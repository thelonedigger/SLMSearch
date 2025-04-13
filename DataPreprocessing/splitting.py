import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm
import argparse
import sys
import math

#===============================================================================
# DEFAULT CONFIGURATION (Edit these values if not using command-line arguments)
#===============================================================================
# Path to directory containing MS MARCO files
DEFAULT_DATA_DIR = "./msmarco"  # Change this to your actual data directory path
# Path to save processed files
DEFAULT_OUTPUT_DIR = "./processed_data"
# Whether to create custom splits instead of using standard MS MARCO splits
DEFAULT_USE_CUSTOM_SPLITS = True
# Ratio of data to use for validation (only for custom splits)
DEFAULT_VAL_RATIO = 0.1
# Ratio of data to use for testing (only for custom splits)
DEFAULT_TEST_RATIO = 0.1
# Whether to create training examples with positive and negative passages
DEFAULT_CREATE_EXAMPLES = True
# Number of negative passages per positive example
DEFAULT_N_NEGATIVES = 25
# Maximum number of queries to use for creating examples (None = use all)
DEFAULT_MAX_EXAMPLES = None
# Random seed
DEFAULT_SEED = 12
# Which chunk to process (1-based index)
DEFAULT_CHUNK = 1
# Total number of chunks to split processing into
DEFAULT_TOTAL_CHUNKS = 2000
# Whether this is the final chunk that should merge all previous chunks
DEFAULT_MERGE_CHUNKS = False

"""
What this script does:

1. Loads a chunk of the MS MARCO dataset files
2. Preprocesses text data (lowercase, whitespace normalization)
3. Creates data splits:
   - Either uses MS MARCO's default train/dev splits
   - Or creates custom train/validation/test splits
4. Optionally creates training examples with:
   - Query
   - Positive (relevant) passage
   - Multiple negative (non-relevant) passages
5. Saves processed data in pickle format for efficient loading

Output files (per chunk):
- processed_data/chunk{i}/collection.pkl: Subset of passages with original and processed text
- processed_data/chunk{i}/train_queries.pkl: Training queries relevant to this chunk
- processed_data/chunk{i}/train_qrels.pkl: Training relevance judgments relevant to this chunk
- processed_data/chunk{i}/val_queries.pkl: Validation queries relevant to this chunk
- processed_data/chunk{i}/val_qrels.pkl: Validation relevance judgments relevant to this chunk
- processed_data/chunk{i}/train_examples.pkl: Training examples with positive and negative passages

Final merged files:
- processed_data/collection.pkl: All passages with original and processed text
- processed_data/train_queries.pkl: All training queries
- processed_data/train_qrels.pkl: All training relevance judgments
- processed_data/val_queries.pkl: All validation queries
- processed_data/val_qrels.pkl: All validation relevance judgments
- processed_data/train_examples.pkl: All training examples
"""

def load_msmarco_data_chunked(data_dir, chunk, total_chunks):
    """
    Load a chunk of MS MARCO dataset files
    """
    print(f"Loading MS MARCO files (Chunk {chunk}/{total_chunks})...")
    
    # Load collection (passages) - only the specific chunk
    collection_path = os.path.join(data_dir, 'collection.tsv')
    
    # Count total lines in collection file to determine chunk size
    with open(collection_path, 'r', encoding='utf-8') as f:
        total_passages = sum(1 for _ in f)
    
    chunk_size = math.ceil(total_passages / total_chunks)
    start_line = (chunk - 1) * chunk_size
    end_line = min(chunk * chunk_size, total_passages)
    
    print(f"Total passages: {total_passages}")
    print(f"Processing passages from line {start_line} to {end_line} (Chunk size: {chunk_size})")
    
    # Read only the specific chunk of passages
    collection_df = pd.DataFrame(columns=['pid', 'passage'])
    with open(collection_path, 'r', encoding='utf-8') as f:
        # Use tqdm to track progress through all lines but maintain original line counting logic
        for i, line in tqdm(enumerate(f), total=end_line, desc="Loading passages"):
            # Skip lines before our chunk starts
            if i < start_line:
                continue
            
            # Stop when we reach the end of our chunk
            if i >= end_line:
                break
            
            parts = line.strip().split('\t')
            if len(parts) == 2:
                pid, passage = parts
                collection_df = pd.concat([collection_df, pd.DataFrame({'pid': [pid], 'passage': [passage]})], ignore_index=True)
            
            # Keep the original progress reporting behavior for consistency
            if (i - start_line + 1) % 100000 == 0:
                print(f"Loaded {i - start_line + 1} passages...")
    
    print(f"Loaded {len(collection_df)} passages for this chunk")
    
    # Get the passage IDs in this chunk
    passage_ids = set(collection_df['pid'].values)
    
    # Load all queries (we'll filter them later based on qrels)
    train_queries_path = os.path.join(data_dir, 'queries.train.tsv')
    dev_queries_path = os.path.join(data_dir, 'queries.dev.small.tsv')
    
    train_queries_df = pd.read_csv(train_queries_path, sep='\t', header=None, names=['qid', 'query'])
    dev_queries_df = pd.read_csv(dev_queries_path, sep='\t', header=None, names=['qid', 'query'])
    
    print(f"Loaded {len(train_queries_df)} training queries and {len(dev_queries_df)} dev queries")
    
    # Load relevance judgments and filter to those relevant to this chunk
    train_qrels_path = os.path.join(data_dir, 'qrels.train.tsv')
    dev_qrels_path = os.path.join(data_dir, 'qrels.dev.small.tsv')
    
    train_qrels_df = pd.read_csv(train_qrels_path, sep='\t', header=None, names=['qid', 'unused', 'pid', 'relevance'])
    dev_qrels_df = pd.read_csv(dev_qrels_path, sep='\t', header=None, names=['qid', 'unused', 'pid', 'relevance'])
    
    # Add diagnostic information
    print(f"Sample passage IDs from collection: {list(passage_ids)[:5]}")
    print(f"Sample passage IDs from qrels: {train_qrels_df['pid'].iloc[:5].tolist()}")
    print(f"Types - collection: {type(list(passage_ids)[0]) if passage_ids else 'N/A'}")
    print(f"Types - qrels: {type(train_qrels_df['pid'].iloc[0]) if len(train_qrels_df) > 0 else 'N/A'}")
    
    # Fix type mismatch
    train_qrels_df['pid'] = train_qrels_df['pid'].astype(str)
    dev_qrels_df['pid'] = dev_qrels_df['pid'].astype(str)
    passage_ids_str = set(str(pid) for pid in passage_ids)
    
    # Filter qrels to only those that reference passages in this chunk
    train_qrels_df = train_qrels_df[train_qrels_df['pid'].isin(passage_ids_str)]
    dev_qrels_df = dev_qrels_df[dev_qrels_df['pid'].isin(passage_ids_str)]
    
    print(f"Filtered to {len(train_qrels_df)} training relevance judgments and {len(dev_qrels_df)} dev relevance judgments for this chunk")
    
    # Filter queries to only those that appear in the filtered qrels
    train_query_ids = set(train_qrels_df['qid'].values)
    dev_query_ids = set(dev_qrels_df['qid'].values)
    
    train_queries_df = train_queries_df[train_queries_df['qid'].isin(train_query_ids)]
    dev_queries_df = dev_queries_df[dev_queries_df['qid'].isin(dev_query_ids)]
    
    print(f"Filtered to {len(train_queries_df)} training queries and {len(dev_queries_df)} dev queries for this chunk")
    
    return collection_df, train_queries_df, dev_queries_df, train_qrels_df, dev_qrels_df

def preprocess_text(text):
    """
    Basic text preprocessing function
    """
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove excessive whitespace
        text = ' '.join(text.split())
        return text
    return ""

def create_custom_splits(queries_df, qrels_df, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Create custom train/validation/test splits from the data
    """
    print("Creating custom train/val/test splits...")
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Get unique query IDs that have relevance judgments
    unique_qids = qrels_df['qid'].unique()
    np.random.shuffle(unique_qids)
    
    # Split query IDs
    n_total = len(unique_qids)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test
    
    train_qids = unique_qids[:n_train]
    val_qids = unique_qids[n_train:n_train+n_val]
    test_qids = unique_qids[n_train+n_val:]
    
    print(f"Split into {len(train_qids)} train, {len(val_qids)} validation, and {len(test_qids)} test queries")
    
    # Create splits
    train_queries = queries_df[queries_df['qid'].isin(train_qids)]
    train_qrels = qrels_df[qrels_df['qid'].isin(train_qids)]
    
    val_queries = queries_df[queries_df['qid'].isin(val_qids)]
    val_qrels = qrels_df[qrels_df['qid'].isin(val_qids)]
    
    test_queries = queries_df[queries_df['qid'].isin(test_qids)]
    test_qrels = qrels_df[qrels_df['qid'].isin(test_qids)]
    
    return (train_queries, train_qrels), (val_queries, val_qrels), (test_queries, test_qrels)

def create_training_examples(queries_df, qrels_df, collection_df, n_negatives=5, max_samples=None):
    """
    Create training examples with positive and negative passages for each query
    """
    print(f"Creating training examples with {n_negatives} negative samples per positive...")
    
    examples = []
    
    # Get all passage IDs for sampling negatives
    all_pids = set(collection_df['pid'].values)
    
    # Group qrels by query ID
    qrels_grouped = qrels_df.groupby('qid')
    
    # If max_samples is set, limit the number of queries
    query_list = list(zip(queries_df['qid'], queries_df['processed_query']))
    if max_samples and max_samples < len(query_list):
        query_list = random.sample(query_list, max_samples)
    
    for qid, query in tqdm(query_list, total=len(query_list)):
        if qid in qrels_grouped.groups:
            # Get positive passage IDs for this query
            pos_pids = qrels_grouped.get_group(qid)['pid'].values
            
            for pos_pid in pos_pids:
                # Get positive passage text
                pos_passage = collection_df.loc[collection_df['pid'] == pos_pid, 'processed_passage'].values
                if len(pos_passage) > 0:
                    # Sample negative passages (not relevant to this query)
                    neg_pid_pool = list(all_pids - set(pos_pids))
                    if len(neg_pid_pool) >= n_negatives:
                        neg_pids = random.sample(neg_pid_pool, n_negatives)
                        neg_passages = []
                        
                        for neg_pid in neg_pids:
                            neg_passage = collection_df.loc[collection_df['pid'] == neg_pid, 'processed_passage'].values
                            if len(neg_passage) > 0:
                                neg_passages.append((neg_pid, neg_passage[0]))
                        
                        # Only add if we got all negative passages
                        if len(neg_passages) == n_negatives:
                            # Add example
                            examples.append({
                                'qid': qid,
                                'query': query,
                                'pos_pid': pos_pid,
                                'pos_passage': pos_passage[0],
                                'neg_examples': neg_passages
                            })
    
    print(f"Created {len(examples)} training examples")
    return pd.DataFrame(examples)

def save_processed_data(output_dir, collection_df, train_data, val_data, test_data=None, train_examples=None, chunk=None):
    """
    Save all processed data to disk
    """
    if chunk is not None:
        # Create a chunk-specific subdirectory
        chunk_dir = os.path.join(output_dir, f'chunk{chunk}')
        print(f"Saving processed data for chunk {chunk} to {chunk_dir}...")
        os.makedirs(chunk_dir, exist_ok=True)
        
        # Save files without suffix since they're in a chunk-specific directory
        collection_df.to_pickle(os.path.join(chunk_dir, 'collection.pkl'))
        
        # Save train/val/test splits
        train_queries, train_qrels = train_data
        val_queries, val_qrels = val_data
        
        train_queries.to_pickle(os.path.join(chunk_dir, 'train_queries.pkl'))
        train_qrels.to_pickle(os.path.join(chunk_dir, 'train_qrels.pkl'))
        val_queries.to_pickle(os.path.join(chunk_dir, 'val_queries.pkl'))
        val_qrels.to_pickle(os.path.join(chunk_dir, 'val_qrels.pkl'))
        
        # Save test data if provided
        if test_data:
            test_queries, test_qrels = test_data
            test_queries.to_pickle(os.path.join(chunk_dir, 'test_queries.pkl'))
            test_qrels.to_pickle(os.path.join(chunk_dir, 'test_qrels.pkl'))
        
        # Save training examples if created
        if train_examples is not None:
            train_examples.to_pickle(os.path.join(chunk_dir, 'train_examples.pkl'))
    else:
        # Merged data goes directly to the output directory
        print(f"Saving merged processed data to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save processed collection
        collection_df.to_pickle(os.path.join(output_dir, 'collection.pkl'))
        
        # Save train/val/test splits
        train_queries, train_qrels = train_data
        val_queries, val_qrels = val_data
        
        train_queries.to_pickle(os.path.join(output_dir, 'train_queries.pkl'))
        train_qrels.to_pickle(os.path.join(output_dir, 'train_qrels.pkl'))
        val_queries.to_pickle(os.path.join(output_dir, 'val_queries.pkl'))
        val_qrels.to_pickle(os.path.join(output_dir, 'val_qrels.pkl'))
        
        # Save test data if provided
        if test_data:
            test_queries, test_qrels = test_data
            test_queries.to_pickle(os.path.join(output_dir, 'test_queries.pkl'))
            test_qrels.to_pickle(os.path.join(output_dir, 'test_qrels.pkl'))
        
        # Save training examples if created
        if train_examples is not None:
            train_examples.to_pickle(os.path.join(output_dir, 'train_examples.pkl'))
    
    print("All data saved successfully!")

def merge_processed_chunks(output_dir, total_chunks):
    """
    Merge all processed chunks into a single dataset
    """
    print(f"Merging {total_chunks} chunks from {output_dir}...")
    
    # Initialize empty dataframes for merging
    merged_collection = pd.DataFrame(columns=['pid', 'passage', 'processed_passage'])
    merged_train_queries = pd.DataFrame(columns=['qid', 'query', 'processed_query'])
    merged_train_qrels = pd.DataFrame(columns=['qid', 'unused', 'pid', 'relevance'])
    merged_val_queries = pd.DataFrame(columns=['qid', 'query', 'processed_query'])
    merged_val_qrels = pd.DataFrame(columns=['qid', 'unused', 'pid', 'relevance'])
    merged_train_examples = pd.DataFrame()
    
    # Check which chunks exist
    existing_chunks = []
    for i in range(1, total_chunks + 1):
        chunk_dir = os.path.join(output_dir, f'chunk{i}')
        collection_path = os.path.join(chunk_dir, 'collection.pkl')
        if os.path.exists(collection_path):
            existing_chunks.append(i)
    
    print(f"Found {len(existing_chunks)} existing chunks: {existing_chunks}")
    
    # Merge each chunk
    for i in existing_chunks:
        chunk_dir = os.path.join(output_dir, f'chunk{i}')
        print(f"Loading chunk {i} from {chunk_dir}...")
        
        # Load and merge collection
        collection_path = os.path.join(chunk_dir, 'collection.pkl')
        if os.path.exists(collection_path):
            chunk_collection = pd.read_pickle(collection_path)
            merged_collection = pd.concat([merged_collection, chunk_collection], ignore_index=True)
        
        # Load and merge train queries
        train_queries_path = os.path.join(chunk_dir, 'train_queries.pkl')
        if os.path.exists(train_queries_path):
            chunk_train_queries = pd.read_pickle(train_queries_path)
            merged_train_queries = pd.concat([merged_train_queries, chunk_train_queries], ignore_index=True)
        
        # Load and merge train qrels
        train_qrels_path = os.path.join(chunk_dir, 'train_qrels.pkl')
        if os.path.exists(train_qrels_path):
            chunk_train_qrels = pd.read_pickle(train_qrels_path)
            merged_train_qrels = pd.concat([merged_train_qrels, chunk_train_qrels], ignore_index=True)
        
        # Load and merge val queries
        val_queries_path = os.path.join(chunk_dir, 'val_queries.pkl')
        if os.path.exists(val_queries_path):
            chunk_val_queries = pd.read_pickle(val_queries_path)
            merged_val_queries = pd.concat([merged_val_queries, chunk_val_queries], ignore_index=True)
        
        # Load and merge val qrels
        val_qrels_path = os.path.join(chunk_dir, 'val_qrels.pkl')
        if os.path.exists(val_qrels_path):
            chunk_val_qrels = pd.read_pickle(val_qrels_path)
            merged_val_qrels = pd.concat([merged_val_qrels, chunk_val_qrels], ignore_index=True)
        
        # Load and merge training examples
        train_examples_path = os.path.join(chunk_dir, 'train_examples.pkl')
        if os.path.exists(train_examples_path):
            chunk_train_examples = pd.read_pickle(train_examples_path)
            merged_train_examples = pd.concat([merged_train_examples, chunk_train_examples], ignore_index=True)
    
    # Remove duplicates
    merged_collection = merged_collection.drop_duplicates(subset=['pid'])
    merged_train_queries = merged_train_queries.drop_duplicates(subset=['qid'])
    merged_train_qrels = merged_train_qrels.drop_duplicates(subset=['qid', 'pid'])
    merged_val_queries = merged_val_queries.drop_duplicates(subset=['qid'])
    merged_val_qrels = merged_val_qrels.drop_duplicates(subset=['qid', 'pid'])
    
    print(f"Merged collection size: {len(merged_collection)}")
    print(f"Merged train queries size: {len(merged_train_queries)}")
    print(f"Merged train qrels size: {len(merged_train_qrels)}")
    print(f"Merged val queries size: {len(merged_val_queries)}")
    print(f"Merged val qrels size: {len(merged_val_qrels)}")
    print(f"Merged train examples size: {len(merged_train_examples)}")
    
    # Save merged data
    train_data = (merged_train_queries, merged_train_qrels)
    val_data = (merged_val_queries, merged_val_qrels)
    
    save_processed_data(output_dir, merged_collection, train_data, val_data, None, merged_train_examples)

def main():
    # Parse command-line arguments if provided, otherwise use defaults
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Preprocess MS MARCO dataset in chunks")
        parser.add_argument("--data_dir", type=str, help="Directory containing MS MARCO files")
        parser.add_argument("--output_dir", type=str, help="Output directory for processed files")
        parser.add_argument("--custom_splits", action="store_true", help="Create custom train/val/test splits instead of using standard MS MARCO splits")
        parser.add_argument("--val_ratio", type=float, help="Validation set ratio (for custom splits)")
        parser.add_argument("--test_ratio", type=float, help="Test set ratio (for custom splits)")
        parser.add_argument("--create_examples", action="store_true", help="Create training examples with positive and negative passages")
        parser.add_argument("--n_negatives", type=int, help="Number of negative passages per positive example")
        parser.add_argument("--max_examples", type=int, help="Maximum number of queries to use for creating examples")
        parser.add_argument("--seed", type=int, help="Random seed")
        parser.add_argument("--chunk", type=int, help="Which chunk to process (1-based index)")
        parser.add_argument("--total_chunks", type=int, help="Total number of chunks to split processing into")
        parser.add_argument("--merge", action="store_true", help="Merge all chunks into a single dataset")
        
        args = parser.parse_args()
        
        # Use provided arguments or fall back to defaults
        data_dir = args.data_dir if args.data_dir is not None else DEFAULT_DATA_DIR
        output_dir = args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_DIR
        custom_splits = args.custom_splits if args.custom_splits is not None else DEFAULT_USE_CUSTOM_SPLITS
        val_ratio = args.val_ratio if args.val_ratio is not None else DEFAULT_VAL_RATIO
        test_ratio = args.test_ratio if args.test_ratio is not None else DEFAULT_TEST_RATIO
        create_examples = args.create_examples if args.create_examples is not None else DEFAULT_CREATE_EXAMPLES
        n_negatives = args.n_negatives if args.n_negatives is not None else DEFAULT_N_NEGATIVES
        max_examples = args.max_examples if args.max_examples is not None else DEFAULT_MAX_EXAMPLES
        seed = args.seed if args.seed is not None else DEFAULT_SEED
        chunk = args.chunk if args.chunk is not None else DEFAULT_CHUNK
        total_chunks = args.total_chunks if args.total_chunks is not None else DEFAULT_TOTAL_CHUNKS
        merge_chunks = args.merge if args.merge is not None else DEFAULT_MERGE_CHUNKS
    else:
        # Use default configuration
        print("No command-line arguments provided. Using default configuration.")
        data_dir = DEFAULT_DATA_DIR
        output_dir = DEFAULT_OUTPUT_DIR
        custom_splits = DEFAULT_USE_CUSTOM_SPLITS
        val_ratio = DEFAULT_VAL_RATIO
        test_ratio = DEFAULT_TEST_RATIO
        create_examples = DEFAULT_CREATE_EXAMPLES
        n_negatives = DEFAULT_N_NEGATIVES
        max_examples = DEFAULT_MAX_EXAMPLES
        seed = DEFAULT_SEED
        chunk = DEFAULT_CHUNK
        total_chunks = DEFAULT_TOTAL_CHUNKS
        merge_chunks = DEFAULT_MERGE_CHUNKS
    
    print(f"Configuration:")
    print(f"- Data directory: {data_dir}")
    print(f"- Output directory: {output_dir}")
    print(f"- Using custom splits: {custom_splits}")
    print(f"- Validation ratio: {val_ratio}")
    print(f"- Test ratio: {test_ratio}")
    print(f"- Creating examples: {create_examples}")
    print(f"- Negatives per positive: {n_negatives}")
    print(f"- Max examples: {max_examples}")
    print(f"- Random seed: {seed}")
    print(f"- Chunk: {chunk}/{total_chunks}")
    print(f"- Merge chunks: {merge_chunks}")
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # If merging, merge all chunks and exit
    if merge_chunks:
        merge_processed_chunks(output_dir, total_chunks)
        return
    
    # Load MS MARCO data for this chunk
    collection_df, train_queries_df, dev_queries_df, train_qrels_df, dev_qrels_df = load_msmarco_data_chunked(data_dir, chunk, total_chunks)
    
    # Preprocess text
    print("Preprocessing text...")
    collection_df['processed_passage'] = collection_df['passage'].apply(preprocess_text)
    train_queries_df['processed_query'] = train_queries_df['query'].apply(preprocess_text)
    dev_queries_df['processed_query'] = dev_queries_df['query'].apply(preprocess_text)
    
    # Create splits
    if custom_splits:
        # Combine train and dev data for custom splitting
        all_queries_df = pd.concat([train_queries_df, dev_queries_df]).drop_duplicates(subset=['qid'])
        all_qrels_df = pd.concat([train_qrels_df, dev_qrels_df]).drop_duplicates(subset=['qid', 'pid'])
        
        # Create custom splits
        train_data, val_data, test_data = create_custom_splits(
            all_queries_df, all_qrels_df, 
            val_ratio=val_ratio, 
            test_ratio=test_ratio,
            seed=seed
        )
    else:
        # Use standard MS MARCO splits
        print("Using standard MS MARCO splits...")
        train_data = (train_queries_df, train_qrels_df)
        val_data = (dev_queries_df, dev_qrels_df)
        test_data = None  # No labeled test data in MS MARCO
    
    # Create training examples
    train_examples = None
    if create_examples:
        train_queries, train_qrels = train_data
        train_examples = create_training_examples(
            train_queries, train_qrels, collection_df,
            n_negatives=n_negatives,
            max_samples=max_examples
        )
    
    # Save processed data for this chunk
    save_processed_data(
        output_dir,
        collection_df,
        train_data,
        val_data,
        test_data,
        train_examples,
        chunk
    )

if __name__ == "__main__":
    main()