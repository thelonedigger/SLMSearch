"""
Embedding Engine for Semantic Document Retrieval

This module provides functionality for generating embeddings from text
using the MiniLM model from the sentence-transformers library. It also
implements efficient vector storage and similarity search using FAISS.

Classes:
    EmbeddingEngine: Main class for embedding generation and vector storage
    
Functions:
    create_embedding_engine: Factory function to create a configured EmbeddingEngine
"""

import os
import numpy as np
import torch
import faiss
import pickle
from typing import List, Tuple, Dict, Optional, Union, Any, Callable
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time
import asyncio
import concurrent.futures

class EmbeddingEngine:
    """
    Engine for generating embeddings and performing similarity search.
    
    This class handles the embedding model, vector storage, and similarity search 
    for semantic document retrieval.
    
    Attributes:
        model (SentenceTransformer): Sentence transformer model for generating embeddings
        index (faiss.Index): FAISS index for efficient similarity search
        passage_ids (List[str]): List of passage IDs in the index
        device (str): Device to run the model on ('cuda' or 'cpu')
        dimension (int): Dimension of the embeddings
        index_initialized (bool): Whether the index has been initialized
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', use_gpu: bool = False):
        """
        Initialize the embedding engine.
        
        Args:
            model_name (str): Name of the sentence-transformers model to use
            use_gpu (bool): Whether to use GPU for embedding generation
        """
        # Set device
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Get embedding dimension
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.dimension}")
        
        # Initialize FAISS index
        self.index = None
        self.passage_ids = []
        self.index_initialized = False
        
        # Progress tracking
        self.progress_callback = None
        self.cancellation_check = None

    # Add a helper function to safely run a coroutine from a non-async context
    def _run_progress_callback(self, progress, message, callback):
        """Safely run an async progress callback from a non-async context"""
        if not callback:
            return False
        
        try:
            # Create a coroutine that checks if the callback returns a coroutine
            async def callback_coro():
                try:
                    result = callback(progress, message)
                    if asyncio.iscoroutine(result):
                        # If it's a coroutine, await it
                        return await result
                    else:
                        # If it's a direct value, return it
                        return result
                except Exception as e:
                    print(f"Error in progress callback coroutine: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
                    return False
                    
            # Run in the event loop
            loop = asyncio.get_event_loop()
            future = asyncio.run_coroutine_threadsafe(callback_coro(), loop)
            
            # Wait with timeout and handle errors
            try:
                return future.result(timeout=5)
            except concurrent.futures.TimeoutError:
                print("Progress callback timed out (non-critical)")
                return False
            except Exception as e:
                print(f"Error in progress callback future: {str(e)}")
                import traceback
                print(traceback.format_exc())
                return False
        except Exception as e:
            print(f"Error setting up progress callback: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query into an embedding vector.
        
        Args:
            query (str): Query text
            
        Returns:
            np.ndarray: Query embedding vector
        """
        with torch.no_grad():
            embedding = self.model.encode(query, convert_to_numpy=True, show_progress_bar=False)
        return embedding
    
    def encode_queries(self, queries: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of queries into embedding vectors.
        
        Args:
            queries (List[str]): List of query texts
            batch_size (int): Batch size for encoding
            
        Returns:
            np.ndarray: Array of query embedding vectors
        """
        with torch.no_grad():
            embeddings = self.model.encode(
                queries, 
                batch_size=batch_size, 
                convert_to_numpy=True, 
                show_progress_bar=True
            )
        return embeddings
    
    def encode_passages(self, 
                        passages: List[str], 
                        batch_size: int = 32, 
                        callback: Optional[Callable[[int, int], bool]] = None) -> np.ndarray:
        """
        Encode a list of passages into embedding vectors with tqdm progress bar.
        
        Args:
            passages (List[str]): List of passage texts
            batch_size (int): Batch size for encoding
            callback: Optional callback(current, total) -> should_cancel
            
        Returns:
            np.ndarray: Array of passage embedding vectors
        """
        # Check if we should use a custom progress tracking
        if callback:
            # Use our own batching logic to provide progress updates
            total_batches = (len(passages) + batch_size - 1) // batch_size
            all_embeddings = []
            
            # Create a tqdm progress bar
            progress_bar = tqdm(
                total=total_batches,
                desc="Encoding passages",
                unit="batch",
                ncols=100
            )
            
            for batch_idx in range(total_batches):
                # Check for cancellation
                if callback(batch_idx, total_batches):
                    progress_bar.close()
                    raise InterruptedError("Encoding cancelled by user")
                
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(passages))
                batch = passages[start_idx:end_idx]
                
                with torch.no_grad():
                    batch_embeddings = self.model.encode(
                        batch, 
                        batch_size=batch_size, 
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
                all_embeddings.append(batch_embeddings)
                
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({"passages": f"{end_idx}/{len(passages)}"})
            
            progress_bar.close()
            
            # Combine batches
            return np.vstack(all_embeddings)
        else:
            # Use the built-in encoding with tqdm
            print(f"Encoding {len(passages)} passages with batch size {batch_size}")
            
            # Create manual batching to use tqdm
            total_batches = (len(passages) + batch_size - 1) // batch_size
            all_embeddings = []
            
            with tqdm(total=total_batches, desc="Encoding passages", unit="batch", ncols=100) as progress_bar:
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(passages))
                    batch = passages[start_idx:end_idx]
                    
                    with torch.no_grad():
                        batch_embeddings = self.model.encode(
                            batch, 
                            batch_size=batch_size, 
                            convert_to_numpy=True,
                            show_progress_bar=False
                        )
                    all_embeddings.append(batch_embeddings)
                    
                    # Update progress bar
                    progress_bar.update(1)
                    progress_bar.set_postfix({"passages": f"{end_idx}/{len(passages)}"})
            
            return np.vstack(all_embeddings)
    
    def initialize_index(self, use_gpu_index: bool = False) -> None:
        """
        Initialize the FAISS index.
        
        Args:
            use_gpu_index (bool): Whether to use GPU for the FAISS index
        """
        # Initialize a flat index (exact search) for baseline
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity when normalized)
        
        # Use GPU for the index if requested and available
        if use_gpu_index and torch.cuda.is_available():
            print("Using GPU for FAISS index")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        self.index_initialized = True
    
    def build_index(self, 
                    passage_ids: List[str], 
                    passages: List[str], 
                    batch_size: int = 32,
                    progress_callback: Optional[Callable[[float, str], bool]] = None) -> None:
        """
        Build the FAISS index from passages with tqdm progress bar.
        
        Args:
            passage_ids (List[str]): List of passage IDs
            passages (List[str]): List of passage texts
            batch_size (int): Batch size for encoding
            progress_callback: Optional callback(progress_percent, message) -> should_cancel
        """
        if not self.index_initialized:
            self.initialize_index()
        
        # Create a tqdm progress bar for the overall process
        main_progress = tqdm(
            total=100,
            desc="Building index",
            unit="%",
            ncols=100,
            position=0,
            leave=True
        )
        
        print(f"Building index with {len(passages)} passages")
        start_time = time.time()
        
        # Store passage IDs for lookup
        self.passage_ids = passage_ids
        
        # Process in batches to avoid memory issues
        total_batches = (len(passages) + batch_size - 1) // batch_size
        
        # Initial progress update
        if progress_callback:
            should_cancel = self._run_progress_callback(0, f"Starting index build for {len(passages)} passages", progress_callback)
            if should_cancel:
                main_progress.close()
                raise InterruptedError("Operation cancelled by user")
        
        main_progress.update(5)  # 5% progress for initialization
        
        # Create a batch tracking callback if we have a progress callback
        if progress_callback:
            def batch_callback(current_batch, total_batches):
                progress = (current_batch / total_batches) * 70  # Encoding takes ~70% of the process
                elapsed = time.time() - start_time
                
                # Ensure we log progress regularly
                if current_batch % max(1, total_batches // 50) == 0 or current_batch == total_batches - 1:
                    main_progress.update(max(0, int(progress - main_progress.n)))
                    main_progress.set_postfix({"batch": f"{current_batch}/{total_batches}"})
                
                # Estimate remaining time
                if current_batch > 0:
                    avg_time_per_batch = elapsed / current_batch
                    remaining_batches = total_batches - current_batch
                    estimated_remaining = avg_time_per_batch * remaining_batches
                    time_message = f"Est. remaining: {int(estimated_remaining // 60)}m {int(estimated_remaining % 60)}s"
                else:
                    time_message = "Estimating time..."
                
                message = f"Processed {current_batch}/{total_batches} batches ({int(progress)}%). {time_message}"
                should_cancel = self._run_progress_callback(5 + progress, message, progress_callback)
                return should_cancel
        else:
            batch_callback = None
        
        try:
            # Generate embeddings with progress tracking
            embeddings = self.encode_passages(
                passages, 
                batch_size=batch_size,
                callback=batch_callback
            )
            
            # Update main progress bar
            main_progress.update(max(0, 75 - main_progress.n))  # Ensure we're at 75%
            
            # Progress update before normalization
            if progress_callback:
                should_cancel = self._run_progress_callback(75, "Normalizing embeddings and adding to index...", progress_callback)
                if should_cancel:
                    main_progress.close()
                    raise InterruptedError("Operation cancelled by user")
            
            # Normalize embeddings for cosine similarity
            main_progress.set_postfix({"status": "Normalizing embeddings"})
            faiss.normalize_L2(embeddings)
            main_progress.update(10)  # Now at 85%
            
            # Add to index
            main_progress.set_postfix({"status": "Adding to index"})
            self.index.add(embeddings)
            main_progress.update(10)  # Now at 95%
            
            # Final progress update
            if progress_callback:
                elapsed = time.time() - start_time
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                message = f"Index built with {self.index.ntotal} vectors in {minutes}m {seconds}s"
                should_cancel = self._run_progress_callback(100, message, progress_callback)
                if should_cancel:
                    main_progress.close()
                    raise InterruptedError("Operation cancelled by user")
            
            main_progress.update(max(0, 100 - main_progress.n))
            main_progress.set_postfix({"status": "Complete"})
            main_progress.close()
            
            print(f"Index built with {self.index.ntotal} vectors in {time.time() - start_time:.2f}s")
        
        except InterruptedError as e:
            main_progress.close()
            print(f"Index building was interrupted: {str(e)}")
            raise
        except Exception as e:
            main_progress.close()
            print(f"Error building index: {str(e)}")
            if progress_callback:
                self._run_progress_callback(0, f"Error building index: {str(e)}", progress_callback)
            raise
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for similar passages using a query.
        
        Args:
            query (str): Query text
            top_k (int): Number of top results to return
            
        Returns:
            List[Tuple[str, float]]: List of (passage_id, similarity_score) tuples
        """
        if not self.index_initialized or self.index.ntotal == 0:
            raise ValueError("Index is not initialized or empty. Please build the index first.")
        
        # Encode the query
        query_embedding = self.encode_query(query)
        
        # Reshape for FAISS
        query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Return passage IDs and scores
        results = [(self.passage_ids[idx], score) for score, idx in zip(scores[0], indices[0]) if idx < len(self.passage_ids)]
        
        return results
    
    def batch_search(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Search for similar passages using multiple queries.
        
        Args:
            queries (List[str]): List of query texts
            top_k (int): Number of top results to return per query
            
        Returns:
            List[List[Tuple[str, float]]]: List of lists of (passage_id, similarity_score) tuples
        """
        if not self.index_initialized or self.index.ntotal == 0:
            raise ValueError("Index is not initialized or empty. Please build the index first.")
        
        # Encode the queries
        query_embeddings = self.encode_queries(queries)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embeddings)
        
        # Search the index
        scores, indices = self.index.search(query_embeddings, top_k)
        
        # Return passage IDs and scores for each query
        results = []
        for i in range(len(queries)):
            query_results = [(self.passage_ids[idx], score) for score, idx in zip(scores[i], indices[i]) if idx < len(self.passage_ids)]
            results.append(query_results)
        
        return results
    
    def save_index(self, 
                  save_dir: str, 
                  progress_callback: Optional[Callable[[float, str], bool]] = None) -> None:
        """
        Save the FAISS index and passage IDs to disk.
        
        Args:
            save_dir (str): Directory to save the index
            progress_callback: Optional callback(progress_percent, message) -> should_cancel
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if progress_callback:
            should_cancel = self._run_progress_callback(0, "Starting index save", progress_callback)
            if should_cancel:
                raise InterruptedError("Operation cancelled by user")
        
        # Save the index
        index_path = os.path.join(save_dir, 'faiss_index.bin')
        
        # If the index is on GPU, move it to CPU first
        index_to_save = faiss.index_gpu_to_cpu(self.index) if hasattr(self.index, 'getDevice') else self.index
        
        if progress_callback:
            should_cancel = self._run_progress_callback(30, "Saving FAISS index", progress_callback)
            if should_cancel:
                raise InterruptedError("Operation cancelled by user")
        
        faiss.write_index(index_to_save, index_path)
        print(f"Index saved to {index_path}")
        
        if progress_callback:
            should_cancel = self._run_progress_callback(70, "Saving passage IDs", progress_callback)
            if should_cancel:
                raise InterruptedError("Operation cancelled by user")
        
        # Save passage IDs
        passage_ids_path = os.path.join(save_dir, 'passage_ids.pkl')
        with open(passage_ids_path, 'wb') as f:
            pickle.dump(self.passage_ids, f)
        
        if progress_callback:
            should_cancel = self._run_progress_callback(100, "Index save completed", progress_callback)
            if should_cancel:
                raise InterruptedError("Operation cancelled by user")
        
        print(f"Passage IDs saved to {passage_ids_path}")
    
    def load_index(self, 
                  load_dir: str, 
                  use_gpu_index: bool = False,
                  progress_callback: Optional[Callable[[float, str], bool]] = None) -> None:
        """
        Load the FAISS index and passage IDs from disk.
        
        Args:
            load_dir (str): Directory containing the saved index
            use_gpu_index (bool): Whether to use GPU for the FAISS index
            progress_callback: Optional callback(progress_percent, message) -> should_cancel
        """
        if progress_callback:
            should_cancel = self._run_progress_callback(0, "Starting index load", progress_callback)
            if should_cancel:
                raise InterruptedError("Operation cancelled by user")
        
        # Load passage IDs
        passage_ids_path = os.path.join(load_dir, 'passage_ids.pkl')
        with open(passage_ids_path, 'rb') as f:
            self.passage_ids = pickle.load(f)
        
        if progress_callback:
            should_cancel = self._run_progress_callback(30, f"Loaded {len(self.passage_ids)} passage IDs", progress_callback)
            if should_cancel:
                raise InterruptedError("Operation cancelled by user")
        
        print(f"Loaded {len(self.passage_ids)} passage IDs from {passage_ids_path}")
        
        # Load the index
        index_path = os.path.join(load_dir, 'faiss_index.bin')
        
        if progress_callback:
            should_cancel = self._run_progress_callback(40, "Loading FAISS index", progress_callback)
            if should_cancel:
                raise InterruptedError("Operation cancelled by user")
        
        self.index = faiss.read_index(index_path)
        
        if progress_callback:
            should_cancel = self._run_progress_callback(80, f"Loaded index with {self.index.ntotal} vectors", progress_callback)
            if should_cancel:
                raise InterruptedError("Operation cancelled by user")
        
        print(f"Loaded index with {self.index.ntotal} vectors from {index_path}")
        
        # Move to GPU if requested and available
        if use_gpu_index and torch.cuda.is_available():
            if progress_callback:
                should_cancel = self._run_progress_callback(90, "Moving index to GPU", progress_callback)
                if should_cancel:
                    raise InterruptedError("Operation cancelled by user")
            
            print("Moving index to GPU")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        if progress_callback:
            should_cancel = self._run_progress_callback(100, "Index load completed", progress_callback)
            if should_cancel:
                raise InterruptedError("Operation cancelled by user")
        
        self.index_initialized = True


def create_embedding_engine(model_name: str = 'all-MiniLM-L6-v2', use_gpu: bool = False) -> EmbeddingEngine:
    """
    Factory function to create a configured EmbeddingEngine.
    
    Args:
        model_name (str): Name of the sentence-transformers model to use
        use_gpu (bool): Whether to use GPU for embedding generation
        
    Returns:
        EmbeddingEngine: Configured embedding engine
    """
    return EmbeddingEngine(model_name=model_name, use_gpu=use_gpu)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Embedding Engine for Semantic Document Retrieval")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                        help="Name of the sentence-transformers model to use")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU for embedding generation")
    
    args = parser.parse_args()
    
    # Create embedding engine
    engine = create_embedding_engine(model_name=args.model, use_gpu=args.use_gpu)
    
    # Test with some example texts
    print("\nTesting embedding generation:")
    
    # Example queries
    example_queries = [
        "How to make a cake?",
        "What is the capital of France?",
        "Explain quantum computing"
    ]
    
    # Example passages
    example_passages = [
        "Cake is a form of sweet food made from flour, sugar, and other ingredients, that is usually baked.",
        "Paris is the capital and most populous city of France.",
        "Quantum computing is the use of quantum phenomena such as superposition and entanglement to perform computation."
    ]
    
    # Generate embeddings
    query_embeddings = engine.encode_queries(example_queries)
    passage_embeddings = engine.encode_passages(example_passages)
    
    print(f"Query embeddings shape: {query_embeddings.shape}")
    print(f"Passage embeddings shape: {passage_embeddings.shape}")
    
    # Initialize index with the example passages
    engine.initialize_index()
    engine.build_index(["p1", "p2", "p3"], example_passages)
    
    # Test search
    print("\nTesting search functionality:")
    for query in example_queries:
        results = engine.search(query, top_k=3)
        print(f"Query: {query}")
        for pid, score in results:
            passage_idx = {"p1": 0, "p2": 1, "p3": 2}[pid]
            print(f"  {pid} (score: {score:.4f}): {example_passages[passage_idx]}")
