import os
import pandas as pd

# Directory containing the chunks and where merged files will be saved
OUTPUT_DIR = "./processed_data"
# Total number of chunks to look for
TOTAL_CHUNKS = 20


def merge_processed_chunks(output_dir, total_chunks):
    """
    Merge all processed chunks into a single dataset
    
    Args:
        output_dir (str): Directory containing the chunk subdirectories and where merged files will be saved
        total_chunks (int): Total number of chunks to look for
    """
    print(f"Merging {total_chunks} chunks from {output_dir}...")
    
    # Initialize empty dataframes for merging
    merged_collection = pd.DataFrame(columns=['pid', 'passage', 'processed_passage'])
    merged_train_queries = pd.DataFrame(columns=['qid', 'query', 'processed_query'])
    merged_train_qrels = pd.DataFrame(columns=['qid', 'unused', 'pid', 'relevance'])
    merged_val_queries = pd.DataFrame(columns=['qid', 'query', 'processed_query'])
    merged_val_qrels = pd.DataFrame(columns=['qid', 'unused', 'pid', 'relevance'])
    merged_train_examples = pd.DataFrame()
    merged_test_queries = pd.DataFrame()
    merged_test_qrels = pd.DataFrame()
    
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
            print(f"  - Added {len(chunk_collection)} passages from chunk {i}")
        
        # Load and merge train queries
        train_queries_path = os.path.join(chunk_dir, 'train_queries.pkl')
        if os.path.exists(train_queries_path):
            chunk_train_queries = pd.read_pickle(train_queries_path)
            merged_train_queries = pd.concat([merged_train_queries, chunk_train_queries], ignore_index=True)
            print(f"  - Added {len(chunk_train_queries)} train queries from chunk {i}")
        
        # Load and merge train qrels
        train_qrels_path = os.path.join(chunk_dir, 'train_qrels.pkl')
        if os.path.exists(train_qrels_path):
            chunk_train_qrels = pd.read_pickle(train_qrels_path)
            merged_train_qrels = pd.concat([merged_train_qrels, chunk_train_qrels], ignore_index=True)
            print(f"  - Added {len(chunk_train_qrels)} train qrels from chunk {i}")
        
        # Load and merge val queries
        val_queries_path = os.path.join(chunk_dir, 'val_queries.pkl')
        if os.path.exists(val_queries_path):
            chunk_val_queries = pd.read_pickle(val_queries_path)
            merged_val_queries = pd.concat([merged_val_queries, chunk_val_queries], ignore_index=True)
            print(f"  - Added {len(chunk_val_queries)} val queries from chunk {i}")
        
        # Load and merge val qrels
        val_qrels_path = os.path.join(chunk_dir, 'val_qrels.pkl')
        if os.path.exists(val_qrels_path):
            chunk_val_qrels = pd.read_pickle(val_qrels_path)
            merged_val_qrels = pd.concat([merged_val_qrels, chunk_val_qrels], ignore_index=True)
            print(f"  - Added {len(chunk_val_qrels)} val qrels from chunk {i}")
        
        # Load and merge test data if available
        test_queries_path = os.path.join(chunk_dir, 'test_queries.pkl')
        if os.path.exists(test_queries_path):
            chunk_test_queries = pd.read_pickle(test_queries_path)
            merged_test_queries = pd.concat([merged_test_queries, chunk_test_queries], ignore_index=True)
            print(f"  - Added {len(chunk_test_queries)} test queries from chunk {i}")
            
        test_qrels_path = os.path.join(chunk_dir, 'test_qrels.pkl')
        if os.path.exists(test_qrels_path):
            chunk_test_qrels = pd.read_pickle(test_qrels_path)
            merged_test_qrels = pd.concat([merged_test_qrels, chunk_test_qrels], ignore_index=True)
            print(f"  - Added {len(chunk_test_qrels)} test qrels from chunk {i}")
        
        # Load and merge training examples
        train_examples_path = os.path.join(chunk_dir, 'train_examples.pkl')
        if os.path.exists(train_examples_path):
            chunk_train_examples = pd.read_pickle(train_examples_path)
            merged_train_examples = pd.concat([merged_train_examples, chunk_train_examples], ignore_index=True)
            print(f"  - Added {len(chunk_train_examples)} train examples from chunk {i}")
    
    # Remove duplicates
    print("\nRemoving duplicates from merged data...")
    old_collection_size = len(merged_collection)
    merged_collection = merged_collection.drop_duplicates(subset=['pid'])
    print(f"  - Collection: {old_collection_size} → {len(merged_collection)} passages (removed {old_collection_size - len(merged_collection)} duplicates)")
    
    old_train_queries_size = len(merged_train_queries)
    merged_train_queries = merged_train_queries.drop_duplicates(subset=['qid'])
    print(f"  - Train queries: {old_train_queries_size} → {len(merged_train_queries)} queries (removed {old_train_queries_size - len(merged_train_queries)} duplicates)")
    
    old_train_qrels_size = len(merged_train_qrels)
    merged_train_qrels = merged_train_qrels.drop_duplicates(subset=['qid', 'pid'])
    print(f"  - Train qrels: {old_train_qrels_size} → {len(merged_train_qrels)} relevance judgments (removed {old_train_qrels_size - len(merged_train_qrels)} duplicates)")
    
    old_val_queries_size = len(merged_val_queries)
    merged_val_queries = merged_val_queries.drop_duplicates(subset=['qid'])
    print(f"  - Val queries: {old_val_queries_size} → {len(merged_val_queries)} queries (removed {old_val_queries_size - len(merged_val_queries)} duplicates)")
    
    old_val_qrels_size = len(merged_val_qrels)
    merged_val_qrels = merged_val_qrels.drop_duplicates(subset=['qid', 'pid'])
    print(f"  - Val qrels: {old_val_qrels_size} → {len(merged_val_qrels)} relevance judgments (removed {old_val_qrels_size - len(merged_val_qrels)} duplicates)")
    
    if not merged_test_queries.empty:
        old_test_queries_size = len(merged_test_queries)
        merged_test_queries = merged_test_queries.drop_duplicates(subset=['qid'])
        print(f"  - Test queries: {old_test_queries_size} → {len(merged_test_queries)} queries (removed {old_test_queries_size - len(merged_test_queries)} duplicates)")
    
    if not merged_test_qrels.empty:
        old_test_qrels_size = len(merged_test_qrels)
        merged_test_qrels = merged_test_qrels.drop_duplicates(subset=['qid', 'pid'])
        print(f"  - Test qrels: {old_test_qrels_size} → {len(merged_test_qrels)} relevance judgments (removed {old_test_qrels_size - len(merged_test_qrels)} duplicates)")
    
    # Save merged data
    print("\nSaving merged data to output directory...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed collection
    collection_path = os.path.join(output_dir, 'collection.pkl')
    merged_collection.to_pickle(collection_path)
    print(f"  - Saved collection with {len(merged_collection)} passages to {collection_path}")
    
    # Save train data
    train_queries_path = os.path.join(output_dir, 'train_queries.pkl')
    merged_train_queries.to_pickle(train_queries_path)
    print(f"  - Saved {len(merged_train_queries)} train queries to {train_queries_path}")
    
    train_qrels_path = os.path.join(output_dir, 'train_qrels.pkl')
    merged_train_qrels.to_pickle(train_qrels_path)
    print(f"  - Saved {len(merged_train_qrels)} train qrels to {train_qrels_path}")
    
    # Save val data
    val_queries_path = os.path.join(output_dir, 'val_queries.pkl')
    merged_val_queries.to_pickle(val_queries_path)
    print(f"  - Saved {len(merged_val_queries)} val queries to {val_queries_path}")
    
    val_qrels_path = os.path.join(output_dir, 'val_qrels.pkl')
    merged_val_qrels.to_pickle(val_qrels_path)
    print(f"  - Saved {len(merged_val_qrels)} val qrels to {val_qrels_path}")
    
    # Save test data if available
    if not merged_test_queries.empty:
        test_queries_path = os.path.join(output_dir, 'test_queries.pkl')
        merged_test_queries.to_pickle(test_queries_path)
        print(f"  - Saved {len(merged_test_queries)} test queries to {test_queries_path}")
    
    if not merged_test_qrels.empty:
        test_qrels_path = os.path.join(output_dir, 'test_qrels.pkl')
        merged_test_qrels.to_pickle(test_qrels_path)
        print(f"  - Saved {len(merged_test_qrels)} test qrels to {test_qrels_path}")
    
    # Save training examples if available
    if not merged_train_examples.empty:
        train_examples_path = os.path.join(output_dir, 'train_examples.pkl')
        merged_train_examples.to_pickle(train_examples_path)
        print(f"  - Saved {len(merged_train_examples)} train examples to {train_examples_path}")
    
    print("\nMerge completed successfully!")


if __name__ == "__main__":
    print(f"Configuration:")
    print(f"- Output directory: {OUTPUT_DIR}")
    print(f"- Total chunks: {TOTAL_CHUNKS}")
    
    # Merge all chunks
    merge_processed_chunks(OUTPUT_DIR, TOTAL_CHUNKS)