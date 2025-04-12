"""
Semantic Document Retrieval Pipeline

This module implements the end-to-end retrieval pipeline, coordinating
the data handler and embedding engine to perform retrieval and evaluation.

Classes:
    RetrievalPipeline: Main class for the retrieval pipeline
    
Functions:
    run_evaluation: Run evaluation on a validation set
    calculate_metrics: Calculate retrieval metrics (MRR, nDCG, Precision@k)
"""

import os
import time
import pickle
import argparse
import numpy as np
from typing import List, Tuple, Dict, Set, Optional, Union, Any
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

# Import from other modules
from datapipeline.data_handler import MSMarcoDataset, load_preprocessed_data
from datapipeline.embedding_engine import EmbeddingEngine, create_embedding_engine


class RetrievalPipeline:
    """
    End-to-end retrieval pipeline for semantic document retrieval.
    
    This class coordinates the data handler and embedding engine to
    perform retrieval and evaluation.
    
    Attributes:
        dataset (MSMarcoDataset): Dataset handler
        embedding_engine (EmbeddingEngine): Embedding engine
        index_built (bool): Whether the index has been built
    """
    
    def __init__(self, dataset: MSMarcoDataset, embedding_engine: EmbeddingEngine):
        """
        Initialize the retrieval pipeline.
        
        Args:
            dataset (MSMarcoDataset): Dataset handler
            embedding_engine (EmbeddingEngine): Embedding engine
        """
        self.dataset = dataset
        self.embedding_engine = embedding_engine
        self.index_built = False
    
    def build_index(self, batch_size: int = 32) -> None:
        """
        Build the search index from passages in the dataset.
        
        Args:
            batch_size (int): Batch size for processing
        """
        print("Building search index...")
        
        # Get all passages
        passage_tuples = self.dataset.get_all_passages()
        passage_ids = [pid for pid, _ in passage_tuples]
        passage_texts = [text for _, text in passage_tuples]
        
        # Build the index
        self.embedding_engine.build_index(passage_ids, passage_texts, batch_size=batch_size)
        self.index_built = True
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float, str]]:
        """
        Retrieve passages for a query.
        
        Args:
            query (str): Query text
            top_k (int): Number of top results to return
            
        Returns:
            List[Tuple[str, float, str]]: List of (passage_id, score, passage_text) tuples
        """
        if not self.index_built:
            raise ValueError("Index not built. Please call build_index() first.")
        
        # Search the index
        results = self.embedding_engine.search(query, top_k=top_k)
        
        # Add passage text to results
        detailed_results = []
        for pid, score in results:
            try:
                passage_text = self.dataset.get_passage_text(pid)
                detailed_results.append((pid, score, passage_text))
            except KeyError:
                # Skip passages that are not found in the dataset
                continue
        
        return detailed_results
    
    def batch_retrieve(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[str, float, str]]]:
        """
        Retrieve passages for multiple queries.
        
        Args:
            queries (List[str]): List of query texts
            top_k (int): Number of top results to return per query
            
        Returns:
            List[List[Tuple[str, float, str]]]: List of lists of (passage_id, score, passage_text) tuples
        """
        if not self.index_built:
            raise ValueError("Index not built. Please call build_index() first.")
        
        # Search the index
        batch_results = self.embedding_engine.batch_search(queries, top_k=top_k)
        
        # Add passage text to results
        detailed_batch_results = []
        for results in batch_results:
            detailed_results = []
            for pid, score in results:
                try:
                    passage_text = self.dataset.get_passage_text(pid)
                    detailed_results.append((pid, score, passage_text))
                except KeyError:
                    # Skip passages that are not found in the dataset
                    continue
            detailed_batch_results.append(detailed_results)
        
        return detailed_batch_results
    
    def evaluate(self, split: str = 'val', top_k: int = 100, num_samples: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate the retrieval pipeline on a dataset split.
        
        Args:
            split (str): Dataset split ('train', 'val', or 'test')
            top_k (int): Number of top results to evaluate
            num_samples (Optional[int]): Number of queries to sample for evaluation (None for all)
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        print(f"Evaluating on {split} split...")
        
        # Get queries based on the split
        if split == 'train':
            query_tuples = self.dataset.get_train_queries()
        elif split == 'val':
            query_tuples = self.dataset.get_val_queries()
        elif split == 'test':
            # Check if test split is available
            if not self.dataset.has_test_split():
                raise ValueError("Test split not available. Make sure you're using a custom split dataset.")
            query_tuples = self.dataset.get_test_queries()
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        # Sample queries if needed
        if num_samples is not None and num_samples < len(query_tuples):
            import random
            query_tuples = random.sample(query_tuples, num_samples)
        
        # Prepare query data
        query_ids = [qid for qid, _ in query_tuples]
        query_texts = [text for _, text in query_tuples]
        
        # Retrieve passages
        start_time = time.time()
        all_results = self.batch_retrieve(query_texts, top_k=top_k)
        end_time = time.time()
        
        # Calculate metrics
        metrics = calculate_metrics(query_ids, all_results, self.dataset, split)
        
        # Add timing information
        metrics['retrieval_time'] = end_time - start_time
        metrics['queries_per_second'] = len(query_ids) / (end_time - start_time)
        
        return metrics
    
    def save(self, save_dir: str) -> None:
        """
        Save the retrieval pipeline to disk.
        
        Args:
            save_dir (str): Directory to save the pipeline
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the embedding engine
        self.embedding_engine.save_index(save_dir)
        
        # Save the pipeline state
        state = {
            'index_built': self.index_built
        }
        
        state_path = os.path.join(save_dir, 'pipeline_state.pkl')
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Retrieval pipeline saved to {save_dir}")
    
    def load(self, load_dir: str, use_gpu_index: bool = False) -> None:
        """
        Load the retrieval pipeline from disk.
        
        Args:
            load_dir (str): Directory containing the saved pipeline
            use_gpu_index (bool): Whether to use GPU for the FAISS index
        """
        # Load the embedding engine
        self.embedding_engine.load_index(load_dir, use_gpu_index=use_gpu_index)
        
        # Load the pipeline state
        state_path = os.path.join(load_dir, 'pipeline_state.pkl')
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
        
        self.index_built = state['index_built']
        
        print(f"Retrieval pipeline loaded from {load_dir}")
    
    def plot_evaluation_results(self, metrics: Dict[str, float], save_path: Optional[str] = None) -> None:
        """
        Plot evaluation results.
        
        Args:
            metrics (Dict[str, float]): Dictionary of evaluation metrics
            save_path (Optional[str]): Path to save the plot
        """
        # Extract precision@k metrics
        precision_metrics = {k: v for k, v in metrics.items() if k.startswith('precision_at_')}
        k_values = [int(k.split('_')[-1]) for k in precision_metrics.keys()]
        precision_values = [precision_metrics[f'precision_at_{k}'] for k in k_values]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot precision@k
        plt.subplot(1, 2, 1)
        plt.plot(k_values, precision_values, 'o-', label='Precision@k')
        plt.xlabel('k')
        plt.ylabel('Precision')
        plt.title('Precision@k')
        plt.grid(True)
        
        # Plot other metrics
        plt.subplot(1, 2, 2)
        other_metrics = {k: v for k, v in metrics.items() if not k.startswith('precision_at_') and not k.startswith('retrieval_time') and not k.startswith('queries_per_second')}
        metric_names = list(other_metrics.keys())
        metric_values = list(other_metrics.values())
        
        plt.bar(metric_names, metric_values)
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('Retrieval Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def get_available_splits(self) -> List[str]:
        """
        Get the list of available splits in the dataset.
        
        Returns:
            List[str]: List of available splits ('train', 'val', 'test')
        """
        splits = ['train', 'val']
        if self.dataset.has_test_split():
            splits.append('test')
        return splits


def calculate_metrics(
    query_ids: List[str],
    retrieval_results: List[List[Tuple[str, float, str]]],
    dataset: MSMarcoDataset,
    split: str = 'val'
) -> Dict[str, float]:
    """
    Calculate retrieval metrics.
    
    Args:
        query_ids (List[str]): List of query IDs
        retrieval_results (List[List[Tuple[str, float, str]]]): Retrieval results for each query
        dataset (MSMarcoDataset): Dataset handler
        split (str): Dataset split ('train', 'val', or 'test')
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    # Initialize metrics
    metrics = defaultdict(float)
    
    # Calculate metrics for each query
    for qid, results in zip(query_ids, retrieval_results):
        # Get relevant passages for this query
        relevant_pids = dataset.get_relevant_passages(qid, split=split)
        
        if not relevant_pids:
            # Skip queries with no known relevant passages
            continue
        
        # Extract retrieved passage IDs
        retrieved_pids = [pid for pid, _, _ in results]
        
        # Calculate reciprocal rank (for MRR)
        reciprocal_rank = 0.0
        for i, pid in enumerate(retrieved_pids):
            if pid in relevant_pids:
                reciprocal_rank = 1.0 / (i + 1)
                break
        
        metrics['mrr'] += reciprocal_rank
        
        # Calculate nDCG
        ndcg = calculate_ndcg(retrieved_pids, relevant_pids)
        metrics['ndcg'] += ndcg
        
        # Calculate Precision@k
        for k in [1, 3, 5, 10, 20, 50, 100]:
            if len(retrieved_pids) >= k:
                precision = len(set(retrieved_pids[:k]) & relevant_pids) / k
                metrics[f'precision_at_{k}'] += precision
    
    # Average metrics over all queries
    num_queries = len(query_ids)
    for key in metrics:
        metrics[key] /= num_queries
    
    return dict(metrics)


def calculate_ndcg(retrieved_pids: List[str], relevant_pids: Set[str], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain.
    
    Args:
        retrieved_pids (List[str]): List of retrieved passage IDs
        relevant_pids (Set[str]): Set of relevant passage IDs
        k (int): Cutoff for NDCG calculation
        
    Returns:
        float: NDCG@k value
    """
    # Limit to top k
    retrieved_pids = retrieved_pids[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, pid in enumerate(retrieved_pids):
        if pid in relevant_pids:
            # Binary relevance (1 or 0)
            # Using log base 2 for the discount
            dcg += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed and log(1) = 0
    
    # Calculate ideal DCG
    ideal_dcg = 0.0
    for i in range(min(len(relevant_pids), k)):
        ideal_dcg += 1.0 / np.log2(i + 2)
    
    # Calculate NDCG
    if ideal_dcg > 0:
        ndcg = dcg / ideal_dcg
    else:
        ndcg = 0.0
    
    return ndcg


def run_evaluation(pipeline: RetrievalPipeline, split: str = 'val', top_k: int = 100, 
                  num_samples: Optional[int] = None, plot: bool = True, 
                  plot_path: Optional[str] = None) -> Dict[str, float]:
    """
    Run evaluation on a dataset split.
    
    Args:
        pipeline (RetrievalPipeline): Retrieval pipeline
        split (str): Dataset split ('train', 'val', or 'test')
        top_k (int): Number of top results to evaluate
        num_samples (Optional[int]): Number of queries to sample for evaluation (None for all)
        plot (bool): Whether to plot the results
        plot_path (Optional[str]): Path to save the plot
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    # Check if the split is available
    available_splits = pipeline.get_available_splits()
    if split not in available_splits:
        raise ValueError(f"Split '{split}' not available. Available splits: {available_splits}")
        
    # Evaluate
    metrics = pipeline.evaluate(split=split, top_k=top_k, num_samples=num_samples)
    
    # Print metrics
    print("\nEvaluation Results:")
    for metric, value in sorted(metrics.items()):
        print(f"{metric}: {value:.4f}")
    
    # Plot if requested
    if plot:
        pipeline.plot_evaluation_results(metrics, save_path=plot_path)
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Document Retrieval Pipeline")
    parser.add_argument("--data_dir", type=str, default="./processed_data",
                        help="Directory containing preprocessed MS MARCO data")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2",
                        help="Name of the sentence-transformers model to use")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU for embedding generation")
    parser.add_argument("--save_dir", type=str, default="./saved_pipeline",
                        help="Directory to save the pipeline")
    parser.add_argument("--load_dir", type=str,
                        help="Directory to load the pipeline from")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing")
    parser.add_argument("--eval_split", type=str, default="val", choices=["train", "val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--eval_top_k", type=int, default=100,
                        help="Number of top results to evaluate")
    parser.add_argument("--eval_samples", type=int,
                        help="Number of queries to sample for evaluation")
    parser.add_argument("--query_mode", action="store_true",
                        help="Run in query mode (interactive)")
    
    args = parser.parse_args()
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_preprocessed_data(args.data_dir)
    
    # Create embedding engine
    print("Creating embedding engine...")
    embedding_engine = create_embedding_engine(model_name=args.model_name, use_gpu=args.use_gpu)
    
    # Create retrieval pipeline
    pipeline = RetrievalPipeline(dataset, embedding_engine)
    
    # Load or build the pipeline
    if args.load_dir:
        print(f"Loading pipeline from {args.load_dir}...")
        pipeline.load(args.load_dir, use_gpu_index=args.use_gpu)
    else:
        print("Building search index...")
        pipeline.build_index(batch_size=args.batch_size)
        
        if args.save_dir:
            print(f"Saving pipeline to {args.save_dir}...")
            pipeline.save(args.save_dir)
    
    # Check if the evaluation split is available
    available_splits = pipeline.get_available_splits()
    if args.eval_split not in available_splits:
        print(f"WARNING: Requested split '{args.eval_split}' is not available. Available splits: {available_splits}")
        if 'val' in available_splits:
            args.eval_split = 'val'
            print(f"Falling back to 'val' split instead.")
        else:
            args.eval_split = available_splits[0]
            print(f"Falling back to '{args.eval_split}' split instead.")
    
    # Evaluate
    if not args.query_mode:
        metrics = run_evaluation(
            pipeline, 
            split=args.eval_split, 
            top_k=args.eval_top_k, 
            num_samples=args.eval_samples,
            plot=True,
            plot_path=os.path.join(args.save_dir, f"evaluation_{args.eval_split}.png") if args.save_dir else None
        )
    
    # Query mode
    else:
        print("\nQuery Mode")
        print("Enter a query to search, or 'exit' to quit.")
        
        while True:
            query = input("\nQuery: ")
            
            if query.lower() == 'exit':
                break
            
            # Retrieve passages
            results = pipeline.retrieve(query, top_k=10)
            
            # Print results
            print("\nResults:")
            for i, (pid, score, passage) in enumerate(results):
                print(f"\n{i+1}. {pid} (score: {score:.4f})")
                print(f"   {passage}")