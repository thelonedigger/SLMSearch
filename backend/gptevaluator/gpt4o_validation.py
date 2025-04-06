"""
Validation framework for testing GPT-4o as an evaluator for the semantic retrieval system.
"""

import os
import json
import random
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import prompts from separate file
from evaluation_prompts import get_passage_relevance_prompt, get_passage_ranking_prompt

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def evaluate_with_gpt4o(query, retrieved_passage, gold_passage):
    """
    Evaluate a single passage's relevance to a query using GPT-4o.
    
    Args:
        query: The search query
        retrieved_passage: The passage to evaluate
        gold_passage: A known relevant passage for comparison
        
    Returns:
        dict: GPT-4o's evaluation with relevance score and justification
    """
    # Get the prompt from our prompts file
    prompt = get_passage_relevance_prompt(query, retrieved_passage, gold_passage)
    
    # Call the GPT-4o API
    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except:
        print("Failed to parse JSON response")
        return {"relevance_score": -1, "justification": "Error", "comparison": "Error"}

def evaluate_ranking_with_gpt4o(query, passages, passage_ids):
    """
    Have GPT-4o rank a set of passages for a query.
    
    Args:
        query: The search query
        passages: List of passage texts
        passage_ids: List of passage IDs corresponding to the passages
        
    Returns:
        list: Passage IDs ranked by relevance according to GPT-4o
    """
    # Get the prompt from our prompts file
    prompt = get_passage_ranking_prompt(query, passages)
    
    # Call the GPT-4o API
    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        
        # Get the ranking from the response
        if "ranking" in result:
            ranking = result["ranking"]
        elif isinstance(result, list):
            ranking = result
        else:
            # Try to find a list in the response
            for key, value in result.items():
                if isinstance(value, list):
                    ranking = value
                    break
            else:
                print("Could not find ranking list in response")
                return None
        
        # Map rankings back to passage_ids (adjust for 1-based indexing in prompt)
        ranked_passage_ids = [passage_ids[i-1] for i in ranking]
        return ranked_passage_ids
    except Exception as e:
        print(f"Failed to parse GPT-4o response: {e}")
        print(f"Response was: {response.choices[0].message.content}")
        return None

def load_sample_data(data_dir, num_samples=10):
    """
    Load a sample of queries and passages from MS MARCO.
    
    Args:
        data_dir: Directory containing preprocessed data
        num_samples: Number of queries to sample
        
    Returns:
        list: Sample data for validation
    """
    # Load preprocessed data
    collection = pd.read_pickle(os.path.join(data_dir, 'collection.pkl'))
    val_queries = pd.read_pickle(os.path.join(data_dir, 'val_queries.pkl'))
    val_qrels = pd.read_pickle(os.path.join(data_dir, 'val_qrels.pkl'))
    
    # Get sample queries with known relevant passages
    sample_queries = []
    for _, row in val_queries.sample(num_samples).iterrows():
        qid = row['qid']
        query = row['processed_query']
        
        # Find relevant passages for this query
        relevant_pids = val_qrels[val_qrels['qid'] == qid]['pid'].values
        
        if len(relevant_pids) > 0:
            gold_pid = relevant_pids[0]
            gold_passage = collection[collection['pid'] == gold_pid]['processed_passage'].values[0]
            
            # Get a random passage as a potentially non-relevant one
            random_pid = random.choice(collection['pid'].values)
            while random_pid in relevant_pids:
                random_pid = random.choice(collection['pid'].values)
                
            retrieved_passage = collection[collection['pid'] == random_pid]['processed_passage'].values[0]
            
            sample_queries.append({
                'qid': qid,
                'query': query,
                'gold_pid': gold_pid,
                'gold_passage': gold_passage,
                'retrieved_pid': random_pid,
                'retrieved_passage': retrieved_passage,
                'is_relevant': False  # We know this is likely not relevant
            })
    
    return sample_queries

def validate_gpt4o_evaluation(data_dir, num_samples=10):
    """
    Validate GPT-4o's ability to evaluate passage relevance.
    
    Args:
        data_dir: Directory containing preprocessed data
        num_samples: Number of queries to sample
        
    Returns:
        DataFrame: Validation results
    """
    samples = load_sample_data(data_dir, num_samples)
    results = []
    
    for sample in tqdm(samples, desc="Evaluating passages"):
        print(f"Evaluating query: {sample['query']}")
        
        evaluation = evaluate_with_gpt4o(
            sample['query'], 
            sample['retrieved_passage'], 
            sample['gold_passage']
        )
        
        # Add results
        results.append({
            'qid': sample['qid'],
            'query': sample['query'],
            'gold_pid': sample['gold_pid'],
            'retrieved_pid': sample['retrieved_pid'],
            'is_actually_relevant': sample['is_relevant'],
            'gpt4o_score': evaluation['relevance_score'],
            'gpt4o_justification': evaluation['justification'],
            'gpt4o_comparison': evaluation['comparison']
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('gpt4o_validation_results.csv', index=False)
    
    # Quick analysis
    accuracy = (
        ((results_df['is_actually_relevant'] == True) & (results_df['gpt4o_score'] >= 3)) | 
        ((results_df['is_actually_relevant'] == False) & (results_df['gpt4o_score'] < 3))
    ).mean()
    
    print(f"Preliminary accuracy: {accuracy:.2f}")
    return results_df

def validate_ranking_capability(data_dir, num_queries=10, passages_per_query=5):
    """
    Validate GPT-4o's ability to rank passages compared to ground truth relevance.
    
    Args:
        data_dir: Directory containing preprocessed data
        num_queries: Number of queries to evaluate
        passages_per_query: Number of passages to rank per query
        
    Returns:
        DataFrame: Validation results
    """
    # Load data
    collection = pd.read_pickle(os.path.join(data_dir, 'collection.pkl'))
    val_queries = pd.read_pickle(os.path.join(data_dir, 'val_queries.pkl'))
    val_qrels = pd.read_pickle(os.path.join(data_dir, 'val_qrels.pkl'))
    
    # Results storage
    results = []
    
    # Sample queries that have at least one relevant passage
    query_ids_with_relevance = val_qrels['qid'].unique()
    sample_query_ids = np.random.choice(query_ids_with_relevance, min(num_queries, len(query_ids_with_relevance)), replace=False)
    
    for qid in tqdm(sample_query_ids, desc="Evaluating queries"):
        query_text = val_queries[val_queries['qid'] == qid]['processed_query'].values[0]
        
        # Get relevant passage IDs for this query
        relevant_pids = set(val_qrels[val_qrels['qid'] == qid]['pid'].values)
        
        # Select one relevant passage and (passages_per_query-1) random passages
        if len(relevant_pids) > 0:
            selected_relevant_pid = list(relevant_pids)[0]
            
            # Get random non-relevant passages
            non_relevant_pids = []
            all_pids = set(collection['pid'].values)
            available_pids = list(all_pids - relevant_pids)
            
            if len(available_pids) >= (passages_per_query - 1):
                non_relevant_pids = np.random.choice(available_pids, passages_per_query - 1, replace=False)
            else:
                non_relevant_pids = available_pids
            
            # Combine relevant and non-relevant passage IDs
            selected_pids = [selected_relevant_pid] + list(non_relevant_pids)
            np.random.shuffle(selected_pids)  # Shuffle to randomize position
            
            # Get passage texts
            passages = []
            for pid in selected_pids:
                passage_text = collection[collection['pid'] == pid]['processed_passage'].values
                if len(passage_text) > 0:
                    passages.append(passage_text[0])
                else:
                    passages.append("Passage not found")
            
            # Ground truth ranking (1 for relevant, 0 for non-relevant)
            ground_truth = [1 if pid in relevant_pids else 0 for pid in selected_pids]
            
            # Get GPT-4o ranking
            gpt4o_ranking = evaluate_ranking_with_gpt4o(query_text, passages, selected_pids)
            
            if gpt4o_ranking:
                # Compute metrics (e.g., if relevant passage is ranked first)
                relevant_rank = gpt4o_ranking.index(selected_relevant_pid) + 1 if selected_relevant_pid in gpt4o_ranking else -1
                
                results.append({
                    'qid': qid,
                    'query': query_text,
                    'relevant_pid': selected_relevant_pid,
                    'relevant_rank': relevant_rank,
                    'mean_reciprocal_rank': 1/relevant_rank if relevant_rank > 0 else 0,
                    'selected_pids': selected_pids,
                    'gpt4o_ranking': gpt4o_ranking
                })
    
    # Analyze results
    results_df = pd.DataFrame(results)
    
    # Calculate MRR
    mrr = results_df['mean_reciprocal_rank'].mean()
    
    # Calculate % of times relevant passage is ranked first
    precision_at_1 = (results_df['relevant_rank'] == 1).mean()
    
    print(f"GPT-4o Evaluation Results:")
    print(f"Mean Reciprocal Rank: {mrr:.4f}")
    print(f"Precision@1: {precision_at_1:.4f}")
    
    # Save results
    results_df.to_csv('gpt4o_ranking_validation.csv', index=False)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    rank_counts = results_df['relevant_rank'].value_counts().sort_index()
    rank_counts.plot(kind='bar')
    plt.title('Rank Position of Relevant Passage in GPT-4o Evaluation')
    plt.xlabel('Rank Position')
    plt.ylabel('Count')
    plt.savefig('gpt4o_ranking_results.png')
    
    return results_df

if __name__ == "__main__":
    # Validate both evaluation methods
    print("Starting validation of GPT-4o single passage evaluation...")
    single_results = validate_gpt4o_evaluation("./processed_data", num_samples=10)
    
    print("\nStarting validation of GPT-4o passage ranking...")
    ranking_results = validate_ranking_capability("./processed_data", num_queries=10, passages_per_query=5)