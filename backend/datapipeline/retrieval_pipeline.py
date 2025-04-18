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
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import List, Tuple, Dict, Set, Optional, Union, Any, Callable
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
        status_callback: Async function to report status updates
        cancellation_check: Function to check if an operation was cancelled
        current_operation_id: ID of the current operation
    """
    
    def __init__(self, 
                dataset: MSMarcoDataset, 
                embedding_engine: EmbeddingEngine,
                status_callback: Optional[Callable] = None,
                cancellation_check: Optional[Callable] = None):
        """
        Initialize the retrieval pipeline.
        
        Args:
            dataset (MSMarcoDataset): Dataset handler
            embedding_engine (EmbeddingEngine): Embedding engine
            status_callback: Optional async function for status updates
            cancellation_check: Optional function to check for cancellation
        """
        self.dataset = dataset
        self.embedding_engine = embedding_engine
        self.index_built = False
        self.status_callback = status_callback
        self.cancellation_check = cancellation_check
        self.current_operation_id = None
        
        # Thread pool for running async callbacks from sync contexts
        self._thread_pool = ThreadPoolExecutor(max_workers=1)
    
    async def _update_status(self, status: str, progress: float, details: Optional[str] = None, est_time_remaining: Optional[int] = None):
        """Helper to call status callback if it exists"""
        if self.status_callback and self.current_operation_id:
            await self.status_callback(status, progress, details, est_time_remaining)
    
    # 1. Add this helper function to the class
    def _create_status_update_coro(self, status, progress, details=None, est_time_remaining=None):
        """Create a coroutine for status updates without executing it"""
        async def status_update_coro():
            try:
                if self.status_callback and self.current_operation_id:
                    await self.status_callback(status, progress, details, est_time_remaining)
            except Exception as e:
                print(f"Error in status update coroutine: {str(e)}")
                import traceback
                print(traceback.format_exc())
        return status_update_coro()
    
    def _check_cancelled(self) -> bool:
        """Helper to check if operation was cancelled"""
        if self.cancellation_check:
            return self.cancellation_check()
        return False
    
    def build_index(self, batch_size: int = 32) -> None:
        """
        Build the search index from passages in the dataset.
        
        Args:
            batch_size (int): Batch size for processing
        """
        print("Building search index...")
        
        # Get all passages
        passage_tuples = self.dataset.get_all_passages()
        print(f"Retrieved {len(passage_tuples)} passages for indexing")
        passage_ids = [pid for pid, _ in passage_tuples]
        passage_texts = [text for _, text in passage_tuples]
        
        # Initial status update
        if self.status_callback:
            asyncio.run_coroutine_threadsafe(
                self._update_status("in-progress", 0, f"Starting index build for {len(passage_tuples)} passages"),
                asyncio.get_event_loop()
            )
        
        # 2. Replace the progress_callback implementation in build_index with:
        def progress_callback(progress: float, message: str) -> bool:
            """Progress callback for the embedding engine"""
            # Log progress to console
            if progress % 5 == 0 or progress == 100:
                print(f"Index build progress: {progress:.1f}% - {message}")
                
            # Convert to async and run in thread pool
            if self.status_callback:
                try:
                    # Create a coroutine object without executing it
                    coro = self._create_status_update_coro("in-progress", progress, message)
                    
                    # Run the coroutine in the event loop
                    future = asyncio.run_coroutine_threadsafe(
                        coro,
                        asyncio.get_event_loop()
                    )
                    
                    # Wait for the callback to complete with better error handling
                    try:
                        # Increased timeout to handle slower operations
                        future.result(timeout=5)
                    except Exception as e:
                        import concurrent.futures
                        if isinstance(e, concurrent.futures.TimeoutError):
                            print("Status update timed out (non-critical)")
                        else:
                            print(f"Error in status callback future: {str(e)}")
                            import traceback
                            print(traceback.format_exc())
                except Exception as e:
                    print(f"Error setting up status callback: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
                    
            # Check for cancellation
            return self._check_cancelled()
        
        try:
            # Build the index with progress tracking
            self.embedding_engine.build_index(
                passage_ids, 
                passage_texts, 
                batch_size=batch_size,
                progress_callback=progress_callback
            )
            self.index_built = True
            
            # Final status update if not cancelled
            if not self._check_cancelled() and self.status_callback:
                asyncio.run_coroutine_threadsafe(
                    self._update_status("completed", 100, "Index built successfully"),
                    asyncio.get_event_loop()
                )
        except InterruptedError:
            # Handle cancellation
            print("Index building was cancelled")
            if self.status_callback:
                asyncio.run_coroutine_threadsafe(
                    self._update_status("canceled", 0, "Index building was cancelled"),
                    asyncio.get_event_loop()
                )
        except Exception as e:
            # Handle other errors
            print(f"Error building index: {str(e)}")
            if self.status_callback:
                asyncio.run_coroutine_threadsafe(
                    self._update_status("error", 0, f"Error building index: {str(e)}"),
                    asyncio.get_event_loop()
                )
            raise
    
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
        
        # Update status if callback exists
        if self.status_callback:
            asyncio.run_coroutine_threadsafe(
                self._update_status("in-progress", 20, f"Searching for: {query[:30]}{'...' if len(query) > 30 else ''}"),
                asyncio.get_event_loop()
            )
        
        # Search the index
        results = self.embedding_engine.search(query, top_k=top_k)
        
        # Add passage text to results
        detailed_results = []
        
        # Update status if callback exists
        if self.status_callback:
            asyncio.run_coroutine_threadsafe(
                self._update_status("in-progress", 50, f"Found {len(results)} results, retrieving texts"),
                asyncio.get_event_loop()
            )
        
        for pid, score in results:
            try:
                passage_text = self.dataset.get_passage_text(pid)
                detailed_results.append((pid, score, passage_text))
            except KeyError:
                # Skip passages that are not found in the dataset
                continue
        
        # Final status update
        if self.status_callback:
            asyncio.run_coroutine_threadsafe(
                self._update_status("completed", 100, f"Retrieved {len(detailed_results)} results"),
                asyncio.get_event_loop()
            )
        
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
        
        # Update status with initial progress
        if self.status_callback:
            asyncio.run_coroutine_threadsafe(
                self._update_status("in-progress", 0, f"Starting batch retrieval for {len(queries)} queries"),
                asyncio.get_event_loop()
            )
        
        # Search the index
        batch_results = self.embedding_engine.batch_search(queries, top_k=top_k)
        
        # Add passage text to results
        detailed_batch_results = []
        
        for i, results in enumerate(batch_results):
            # Check for cancellation
            if self._check_cancelled():
                raise InterruptedError("Batch retrieval was cancelled")
            
            # Update progress
            if self.status_callback:
                progress = (i / len(batch_results)) * 100
                asyncio.run_coroutine_threadsafe(
                    self._update_status(
                        "in-progress", 
                        progress, 
                        f"Processing query {i+1}/{len(batch_results)}"
                    ),
                    asyncio.get_event_loop()
                )
            
            detailed_results = []
            for pid, score in results:
                try:
                    passage_text = self.dataset.get_passage_text(pid)
                    detailed_results.append((pid, score, passage_text))
                except KeyError:
                    # Skip passages that are not found in the dataset
                    continue
            detailed_batch_results.append(detailed_results)
        
        # Final status update
        if self.status_callback:
            asyncio.run_coroutine_threadsafe(
                self._update_status("completed", 100, f"Batch retrieval completed for {len(queries)} queries"),
                asyncio.get_event_loop()
            )
        
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
        
        # Initial status update
        if self.status_callback:
            asyncio.run_coroutine_threadsafe(
                self._update_status(
                    "in-progress", 
                    0, 
                    f"Starting evaluation on {split} split with top {top_k} results" +
                    (f" ({num_samples} samples)" if num_samples else "")
                ),
                asyncio.get_event_loop()
            )
        
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
        
        # Status update after query preparation
        if self.status_callback:
            asyncio.run_coroutine_threadsafe(
                self._update_status(
                    "in-progress", 
                    5, 
                    f"Prepared {len(query_tuples)} queries for evaluation"
                ),
                asyncio.get_event_loop()
            )
        
        # Prepare query data
        query_ids = [qid for qid, _ in query_tuples]
        query_texts = [text for _, text in query_tuples]
        
        # Retrieve passages with progress updates
        all_results = []
        start_time = time.time()
        
        try:
            for i, query_text in enumerate(query_texts):
                # Check for cancellation
                if self._check_cancelled():
                    raise InterruptedError("Evaluation was cancelled")
                
                # Update progress
                if self.status_callback:
                    progress = 5 + (i / len(query_texts)) * 70  # 5% to 75%
                    elapsed = time.time() - start_time
                    
                    # Estimate remaining time
                    if i > 0:
                        avg_time_per_query = elapsed / i
                        remaining_queries = len(query_texts) - i
                        est_remaining = int(avg_time_per_query * remaining_queries)
                    else:
                        est_remaining = None
                    
                    asyncio.run_coroutine_threadsafe(
                        self._update_status(
                            "in-progress", 
                            progress, 
                            f"Evaluating query {i+1}/{len(query_texts)}", 
                            est_remaining
                        ),
                        asyncio.get_event_loop()
                    )
                
                # Retrieve for this query
                results = self.embedding_engine.search(query_text, top_k=top_k)
                all_results.append(results)
            
            # Status update before metric calculation
            if self.status_callback:
                asyncio.run_coroutine_threadsafe(
                    self._update_status(
                        "in-progress", 
                        80, 
                        f"Retrieval completed, calculating metrics"
                    ),
                    asyncio.get_event_loop()
                )
            
            # Calculate metrics
            metrics = calculate_metrics(query_ids, all_results, self.dataset, split)
            
            # Add timing information
            end_time = time.time()
            metrics['retrieval_time'] = end_time - start_time
            metrics['queries_per_second'] = len(query_ids) / (end_time - start_time)
            
            # Final status update
            if self.status_callback:
                asyncio.run_coroutine_threadsafe(
                    self._update_status(
                        "completed", 
                        100, 
                        f"Evaluation completed: MRR={metrics.get('mrr', 0):.4f}, nDCG={metrics.get('ndcg', 0):.4f}"
                    ),
                    asyncio.get_event_loop()
                )
            
            return metrics
            
        except InterruptedError:
            # Handle cancellation
            if self.status_callback:
                asyncio.run_coroutine_threadsafe(
                    self._update_status("canceled", 0, "Evaluation was cancelled"),
                    asyncio.get_event_loop()
                )
            raise
        except Exception as e:
            # Handle other errors
            if self.status_callback:
                asyncio.run_coroutine_threadsafe(
                    self._update_status("error", 0, f"Error during evaluation: {str(e)}"),
                    asyncio.get_event_loop()
                )
            raise
    
    def save(self, save_dir: str) -> None:
        """
        Save the retrieval pipeline to disk.
        
        Args:
            save_dir (str): Directory to save the pipeline
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 3. Make similar changes to other methods that use asyncio.run_coroutine_threadsafe:
        # Create a progress callback for the embedding engine
        def progress_callback(progress: float, message: str) -> bool:
            # Convert to async and run in thread pool
            if self.status_callback:
                try:
                    coro = self._create_status_update_coro("in-progress", progress, message)
                    future = asyncio.run_coroutine_threadsafe(
                        coro,
                        asyncio.get_event_loop()
                    )
                    future.result(timeout=5)
                except Exception as e:
                    import concurrent.futures
                    if isinstance(e, concurrent.futures.TimeoutError):
                        print("Status update timed out (non-critical)")
                    else:
                        print(f"Error in status callback future: {str(e)}")
                        import traceback
                        print(traceback.format_exc())
            # Check for cancellation
            return self._check_cancelled()
        
        try:
            # Save the embedding engine with progress tracking
            self.embedding_engine.save_index(save_dir, progress_callback=progress_callback)
            
            # Save the pipeline state
            state = {
                'index_built': self.index_built
            }
            
            state_path = os.path.join(save_dir, 'pipeline_state.pkl')
            with open(state_path, 'wb') as f:
                pickle.dump(state, f)
            
            print(f"Retrieval pipeline saved to {save_dir}")
            
            # Final status update if not cancelled
            if not self._check_cancelled() and self.status_callback:
                asyncio.run_coroutine_threadsafe(
                    self._update_status("completed", 100, f"Pipeline saved to {save_dir}"),
                    asyncio.get_event_loop()
                )
        except InterruptedError:
            # Handle cancellation
            print("Saving was cancelled")
            if self.status_callback:
                asyncio.run_coroutine_threadsafe(
                    self._update_status("canceled", 0, "Saving was cancelled"),
                    asyncio.get_event_loop()
                )
        except Exception as e:
            # Handle other errors
            print(f"Error saving pipeline: {str(e)}")
            if self.status_callback:
                asyncio.run_coroutine_threadsafe(
                    self._update_status("error", 0, f"Error saving pipeline: {str(e)}"),
                    asyncio.get_event_loop()
                )
            raise
    
    def load(self, load_dir: str, use_gpu_index: bool = False) -> None:
        """
        Load the retrieval pipeline from disk.
        
        Args:
            load_dir (str): Directory containing the saved pipeline
            use_gpu_index (bool): Whether to use GPU for the FAISS index
        """
        # 3. Make similar changes to other methods that use asyncio.run_coroutine_threadsafe:
        # Create a progress callback for the embedding engine
        def progress_callback(progress: float, message: str) -> bool:
            # Convert to async and run in thread pool
            if self.status_callback:
                try:
                    coro = self._create_status_update_coro("in-progress", progress, message)
                    future = asyncio.run_coroutine_threadsafe(
                        coro,
                        asyncio.get_event_loop()
                    )
                    future.result(timeout=5)
                except Exception as e:
                    import concurrent.futures
                    if isinstance(e, concurrent.futures.TimeoutError):
                        print("Status update timed out (non-critical)")
                    else:
                        print(f"Error in status callback future: {str(e)}")
                        import traceback
                        print(traceback.format_exc())
            # Check for cancellation
            return self._check_cancelled()
        
        try:
            # Load the embedding engine with progress tracking
            self.embedding_engine.load_index(
                load_dir, 
                use_gpu_index=use_gpu_index,
                progress_callback=progress_callback
            )
            
            # Load the pipeline state
            state_path = os.path.join(load_dir, 'pipeline_state.pkl')
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
            
            self.index_built = state['index_built']
            
            print(f"Retrieval pipeline loaded from {load_dir}")
            
            # Final status update if not cancelled
            if not self._check_cancelled() and self.status_callback:
                asyncio.run_coroutine_threadsafe(
                    self._update_status("completed", 100, f"Pipeline loaded from {load_dir}"),
                    asyncio.get_event_loop()
                )
        except InterruptedError:
            # Handle cancellation
            print("Loading was cancelled")
            if self.status_callback:
                asyncio.run_coroutine_threadsafe(
                    self._update_status("canceled", 0, "Loading was cancelled"),
                    asyncio.get_event_loop()
                )
            raise
        except Exception as e:
            # Handle other errors
            print(f"Error loading pipeline: {str(e)}")
            if self.status_callback:
                asyncio.run_coroutine_threadsafe(
                    self._update_status("error", 0, f"Error loading pipeline: {str(e)}"),
                    asyncio.get_event_loop()
                )
            raise
    
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
    retrieval_results: List[List[Tuple[str, float]]],
    dataset: MSMarcoDataset,
    split: str = 'val'
) -> Dict[str, float]:
    """
    Calculate retrieval metrics.
    
    Args:
        query_ids (List[str]): List of query IDs
        retrieval_results (List[List[Tuple[str, float]]]): Retrieval results for each query
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
        retrieved_pids = [pid for pid, _ in results]
        
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
