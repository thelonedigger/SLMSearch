import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Optional

# Default chunk to check - change this to check a different chunk
DEFAULT_CHUNK = "chunk1"

def load_chunk_data(chunk_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all pickled dataframes from a chunk directory.
    
    Args:
        chunk_dir: Path to the chunk directory
        
    Returns:
        Dict of dataframes with keys corresponding to file names (without .pkl)
    """
    dataframes = {}
    
    # List of expected files
    expected_files = [
        'collection.pkl',
        'train_queries.pkl',
        'train_qrels.pkl',
        'val_queries.pkl',
        'val_qrels.pkl',
        'train_examples.pkl'  # This might not exist if create_examples was false
    ]
    
    for filename in expected_files:
        file_path = os.path.join(chunk_dir, filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_pickle(file_path)
                dataframes[filename.replace('.pkl', '')] = df
                print(f"Loaded {filename} successfully")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"File not found: {file_path}")
    
    return dataframes

def check_data_consistency(dataframes: Dict[str, pd.DataFrame]) -> None:
    """
    Check for basic consistency in the loaded dataframes.
    
    Args:
        dataframes: Dict of dataframes to check
    """
    # Check collection
    if 'collection' in dataframes:
        collection = dataframes['collection']
        print("\n=== Collection ===")
        print(f"Number of passages: {len(collection)}")
        print(f"Columns: {collection.columns.tolist()}")
        print(f"Sample passage:")
        if len(collection) > 0:
            sample = collection.iloc[0]
            print(f"  PID: {sample['pid']}")
            print(f"  Original: {sample['passage'][:100]}...")
            print(f"  Processed: {sample['processed_passage'][:100]}...")
        
        # Check for missing processed text
        missing_processed = collection['processed_passage'].isna().sum()
        if missing_processed > 0:
            print(f"WARNING: {missing_processed} passages have missing processed text")
    
    # Check queries
    for query_type in ['train_queries', 'val_queries']:
        if query_type in dataframes:
            queries = dataframes[query_type]
            print(f"\n=== {query_type.replace('_', ' ').title()} ===")
            print(f"Number of queries: {len(queries)}")
            print(f"Columns: {queries.columns.tolist()}")
            print(f"Sample queries:")
            for _, row in queries.head(3).iterrows():
                print(f"  QID: {row['qid']}")
                print(f"  Query: {row['processed_query']}")
                print()
    
    # Check qrels
    for qrel_type in ['train_qrels', 'val_qrels']:
        if qrel_type in dataframes:
            qrels = dataframes[qrel_type]
            print(f"\n=== {qrel_type.replace('_', ' ').title()} ===")
            print(f"Number of relevance judgments: {len(qrels)}")
            print(f"Columns: {qrels.columns.tolist()}")
            print(f"Sample relevance judgments:")
            for _, row in qrels.head(3).iterrows():
                print(f"  QID: {row['qid']}, PID: {row['pid']}, Relevance: {row['relevance'] if 'relevance' in row else 1}")
    
    # Check training examples
    if 'train_examples' in dataframes:
        examples = dataframes['train_examples']
        print("\n=== Training Examples ===")
        print(f"Number of examples: {len(examples)}")
        print(f"Columns: {examples.columns.tolist()}")
        if len(examples) > 0:
            sample = examples.iloc[0]
            print(f"Sample example:")
            print(f"  QID: {sample['qid']}")
            print(f"  Query: {sample['query']}")
            print(f"  Positive PID: {sample['pos_pid']}")
            print(f"  Positive passage: {sample['pos_passage'][:100]}...")
            if 'neg_examples' in sample and sample['neg_examples']:
                print(f"  Number of negative examples: {len(sample['neg_examples'])}")
                print(f"  First negative example: {sample['neg_examples'][0][1][:100]}...")

def check_reference_integrity(dataframes: Dict[str, pd.DataFrame]) -> None:
    """
    Check if references between dataframes are intact.
    
    Args:
        dataframes: Dict of dataframes to check
    """
    print("\n=== Checking Reference Integrity ===")
    
    # Check if all PIDs in qrels exist in collection
    if 'collection' in dataframes and 'train_qrels' in dataframes:
        collection_pids = set(dataframes['collection']['pid'])
        train_qrel_pids = set(dataframes['train_qrels']['pid'])
        missing_pids = train_qrel_pids - collection_pids
        if missing_pids:
            print(f"WARNING: {len(missing_pids)} PIDs in train_qrels not found in collection")
            print(f"Sample missing PIDs: {list(missing_pids)[:5]}")
        else:
            print("All PIDs in train_qrels exist in collection")
    
    # Check if all QIDs in qrels exist in queries
    if 'train_queries' in dataframes and 'train_qrels' in dataframes:
        train_query_qids = set(dataframes['train_queries']['qid'])
        train_qrel_qids = set(dataframes['train_qrels']['qid'])
        missing_qids = train_qrel_qids - train_query_qids
        if missing_qids:
            print(f"WARNING: {len(missing_qids)} QIDs in train_qrels not found in train_queries")
            print(f"Sample missing QIDs: {list(missing_qids)[:5]}")
        else:
            print("All QIDs in train_qrels exist in train_queries")
    
    # Check val split integrity
    if 'val_queries' in dataframes and 'val_qrels' in dataframes:
        val_query_qids = set(dataframes['val_queries']['qid'])
        val_qrel_qids = set(dataframes['val_qrels']['qid'])
        missing_qids = val_qrel_qids - val_query_qids
        if missing_qids:
            print(f"WARNING: {len(missing_qids)} QIDs in val_qrels not found in val_queries")
            print(f"Sample missing QIDs: {list(missing_qids)[:5]}")
        else:
            print("All QIDs in val_qrels exist in val_queries")
    
    # Check training examples integrity
    if 'train_examples' in dataframes and 'collection' in dataframes:
        collection_pids = set(dataframes['collection']['pid'])
        if 'pos_pid' in dataframes['train_examples'].columns:
            pos_pids = set(dataframes['train_examples']['pos_pid'])
            missing_pids = pos_pids - collection_pids
            if missing_pids:
                print(f"WARNING: {len(missing_pids)} positive PIDs in train_examples not found in collection")
                print(f"Sample missing PIDs: {list(missing_pids)[:5]}")
            else:
                print("All positive PIDs in train_examples exist in collection")

def analyze_data_distribution(dataframes: Dict[str, pd.DataFrame]) -> None:
    """
    Analyze data distribution statistics.
    
    Args:
        dataframes: Dict of dataframes to analyze
    """
    print("\n=== Data Distribution Analysis ===")
    
    # Analyze passages
    if 'collection' in dataframes:
        collection = dataframes['collection']
        
        # Analyze passage length
        if 'processed_passage' in collection.columns:
            passage_lengths = collection['processed_passage'].str.len()
            print(f"Passage length statistics:")
            print(f"  Min: {passage_lengths.min()}")
            print(f"  Max: {passage_lengths.max()}")
            print(f"  Mean: {passage_lengths.mean():.2f}")
            print(f"  Median: {passage_lengths.median()}")
    
    # Analyze queries per QID in train_qrels
    if 'train_qrels' in dataframes:
        train_qrels = dataframes['train_qrels']
        qid_counts = train_qrels['qid'].value_counts()
        print(f"\nRelevant passages per query (train):")
        print(f"  Min: {qid_counts.min()}")
        print(f"  Max: {qid_counts.max()}")
        print(f"  Mean: {qid_counts.mean():.2f}")
        print(f"  Median: {qid_counts.median()}")
        print(f"  Queries with only 1 relevant passage: {(qid_counts == 1).sum()} ({(qid_counts == 1).sum() / len(qid_counts):.2%})")
    
    # Same for validation
    if 'val_qrels' in dataframes:
        val_qrels = dataframes['val_qrels']
        qid_counts = val_qrels['qid'].value_counts()
        print(f"\nRelevant passages per query (validation):")
        print(f"  Min: {qid_counts.min()}")
        print(f"  Max: {qid_counts.max()}")
        print(f"  Mean: {qid_counts.mean():.2f}")
        print(f"  Median: {qid_counts.median()}")
        print(f"  Queries with only 1 relevant passage: {(qid_counts == 1).sum()} ({(qid_counts == 1).sum() / len(qid_counts):.2%})")

def check_chunk_data(chunk_dir: str) -> None:
    """
    Main function to check a chunk's data.
    
    Args:
        chunk_dir: Path to the chunk directory
    """
    print(f"Checking data in {chunk_dir}")
    
    # Load data
    dataframes = load_chunk_data(chunk_dir)
    
    if not dataframes:
        print(f"No data found in {chunk_dir}")
        return
    
    # Check data consistency
    check_data_consistency(dataframes)
    
    # Check reference integrity
    check_reference_integrity(dataframes)
    
    # Analyze data distribution
    analyze_data_distribution(dataframes)
    
    print("\nData check complete!")

if __name__ == "__main__":
    # Check if chunk_dir is provided as command-line argument
    if len(sys.argv) > 1:
        chunk_dir = sys.argv[1]
    else:
        # Use the default chunk from the top of the file
        chunk_dir = f"processed_data/{DEFAULT_CHUNK}"
    
    check_chunk_data(chunk_dir)