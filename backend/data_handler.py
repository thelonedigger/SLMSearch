"""
MS MARCO Dataset Handler

This module provides functionality for loading and managing MS MARCO dataset
that has been preprocessed by the splitting.py script.

Classes:
    MSMarcoDataset: Main class for managing the MS MARCO dataset
    BatchIterator: Iterator for batching dataset elements

Functions:
    load_preprocessed_data: Load preprocessed data from pickle files
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Iterator, Optional, Union, Any


class MSMarcoDataset:
    """
    Handler for the preprocessed MS MARCO dataset.
    
    This class provides functionalities to access queries, passages,
    and relevance judgments from the MS MARCO dataset.
    
    Attributes:
        collection (pd.DataFrame): DataFrame containing passages
        train_queries (pd.DataFrame): DataFrame containing training queries
        train_qrels (pd.DataFrame): DataFrame containing training relevance judgments
        val_queries (pd.DataFrame): DataFrame containing validation queries
        val_qrels (pd.DataFrame): DataFrame containing validation relevance judgments
        passage_id_to_idx (Dict[str, int]): Mapping from passage IDs to indices
        query_id_to_idx (Dict[str, int]): Mapping from query IDs to indices
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the MS MARCO dataset handler.
        
        Args:
            data_dir (str): Directory containing preprocessed MS MARCO data
        """
        self.data_dir = data_dir
        
        # Load the preprocessed data
        self.collection, self.train_queries, self.train_qrels, \
        self.val_queries, self.val_qrels = self._load_data()
        
        # Create mappings for efficient lookups
        self.passage_id_to_idx = {pid: i for i, pid in enumerate(self.collection['pid'])}
        self.train_query_id_to_idx = {qid: i for i, qid in enumerate(self.train_queries['qid'])}
        self.val_query_id_to_idx = {qid: i for i, qid in enumerate(self.val_queries['qid'])}
        
        # Create set of positive passage IDs for each query
        self.train_query_positives = self._create_query_positives(self.train_qrels)
        self.val_query_positives = self._create_query_positives(self.val_qrels)
        
        print(f"Loaded {len(self.collection)} passages")
        print(f"Loaded {len(self.train_queries)} training queries")
        print(f"Loaded {len(self.val_queries)} validation queries")
        print(f"Loaded {len(self.train_qrels)} training relevance judgments")
        print(f"Loaded {len(self.val_qrels)} validation relevance judgments")
    
    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load preprocessed MS MARCO data from pickle files.
        
        Returns:
            Tuple containing:
                - collection: DataFrame with passages
                - train_queries: DataFrame with training queries
                - train_qrels: DataFrame with training relevance judgments
                - val_queries: DataFrame with validation queries
                - val_qrels: DataFrame with validation relevance judgments
        """
        try:
            collection = pd.read_pickle(os.path.join(self.data_dir, 'collection.pkl'))
            train_queries = pd.read_pickle(os.path.join(self.data_dir, 'train_queries.pkl'))
            train_qrels = pd.read_pickle(os.path.join(self.data_dir, 'train_qrels.pkl'))
            val_queries = pd.read_pickle(os.path.join(self.data_dir, 'val_queries.pkl'))
            val_qrels = pd.read_pickle(os.path.join(self.data_dir, 'val_qrels.pkl'))
            
            return collection, train_queries, train_qrels, val_queries, val_qrels
        
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not load preprocessed data: {e}\n"
                                  f"Make sure to run splitting.py first to preprocess the MS MARCO dataset.")
    
    def _create_query_positives(self, qrels: pd.DataFrame) -> Dict[str, Set[str]]:
        """
        Create a dictionary mapping query IDs to sets of positive passage IDs.
        
        Args:
            qrels (pd.DataFrame): DataFrame with relevance judgments
            
        Returns:
            Dict[str, Set[str]]: Mapping from query IDs to sets of positive passage IDs
        """
        query_positives = {}
        
        for _, row in qrels.iterrows():
            qid = row['qid']
            pid = row['pid']
            
            if qid not in query_positives:
                query_positives[qid] = set()
            
            query_positives[qid].add(pid)
        
        return query_positives
    
    def get_passage_text(self, pid: str) -> str:
        """
        Get the text of a passage by its ID.
        
        Args:
            pid (str): Passage ID
            
        Returns:
            str: Processed passage text
        """
        if pid in self.passage_id_to_idx:
            idx = self.passage_id_to_idx[pid]
            return self.collection.iloc[idx]['processed_passage']
        else:
            raise KeyError(f"Passage ID {pid} not found in collection")
    
    def get_train_query_text(self, qid: str) -> str:
        """
        Get the text of a training query by its ID.
        
        Args:
            qid (str): Query ID
            
        Returns:
            str: Processed query text
        """
        if qid in self.train_query_id_to_idx:
            idx = self.train_query_id_to_idx[qid]
            return self.train_queries.iloc[idx]['processed_query']
        else:
            raise KeyError(f"Query ID {qid} not found in training queries")
    
    def get_val_query_text(self, qid: str) -> str:
        """
        Get the text of a validation query by its ID.
        
        Args:
            qid (str): Query ID
            
        Returns:
            str: Processed query text
        """
        if qid in self.val_query_id_to_idx:
            idx = self.val_query_id_to_idx[qid]
            return self.val_queries.iloc[idx]['processed_query']
        else:
            raise KeyError(f"Query ID {qid} not found in validation queries")
    
    def get_relevant_passages(self, qid: str, split: str = 'train') -> Set[str]:
        """
        Get the set of relevant passage IDs for a query.
        
        Args:
            qid (str): Query ID
            split (str): Data split ('train' or 'val')
            
        Returns:
            Set[str]: Set of relevant passage IDs
        """
        if split == 'train':
            if qid in self.train_query_positives:
                return self.train_query_positives[qid]
            else:
                return set()
        elif split == 'val':
            if qid in self.val_query_positives:
                return self.val_query_positives[qid]
            else:
                return set()
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")
    
    def get_all_passages(self) -> List[Tuple[str, str]]:
        """
        Get all passages in the collection.
        
        Returns:
            List[Tuple[str, str]]: List of (passage_id, processed_passage) tuples
        """
        return list(zip(self.collection['pid'], self.collection['processed_passage']))
    
    def get_train_queries(self) -> List[Tuple[str, str]]:
        """
        Get all training queries.
        
        Returns:
            List[Tuple[str, str]]: List of (query_id, processed_query) tuples
        """
        return list(zip(self.train_queries['qid'], self.train_queries['processed_query']))
    
    def get_val_queries(self) -> List[Tuple[str, str]]:
        """
        Get all validation queries.
        
        Returns:
            List[Tuple[str, str]]: List of (query_id, processed_query) tuples
        """
        return list(zip(self.val_queries['qid'], self.val_queries['processed_query']))
    
    def get_passage_batch_iterator(self, batch_size: int = 1000) -> 'BatchIterator':
        """
        Get a batch iterator for passages.
        
        Args:
            batch_size (int): Size of each batch
            
        Returns:
            BatchIterator: Iterator that yields batches of passages
        """
        passages = self.get_all_passages()
        return BatchIterator(passages, batch_size)
    
    def get_train_query_batch_iterator(self, batch_size: int = 100) -> 'BatchIterator':
        """
        Get a batch iterator for training queries.
        
        Args:
            batch_size (int): Size of each batch
            
        Returns:
            BatchIterator: Iterator that yields batches of queries
        """
        queries = self.get_train_queries()
        return BatchIterator(queries, batch_size)
    
    def get_val_query_batch_iterator(self, batch_size: int = 100) -> 'BatchIterator':
        """
        Get a batch iterator for validation queries.
        
        Args:
            batch_size (int): Size of each batch
            
        Returns:
            BatchIterator: Iterator that yields batches of queries
        """
        queries = self.get_val_queries()
        return BatchIterator(queries, batch_size)


class BatchIterator:
    """
    Iterator for batching dataset elements.
    
    This iterator yields batches of elements from the dataset,
    making it easier to process large datasets in smaller chunks.
    
    Attributes:
        data (List): List of data elements to batch
        batch_size (int): Size of each batch
        current_index (int): Current position in the data list
    """
    
    def __init__(self, data: List, batch_size: int):
        """
        Initialize the batch iterator.
        
        Args:
            data (List): List of data elements to batch
            batch_size (int): Size of each batch
        """
        self.data = data
        self.batch_size = batch_size
        self.current_index = 0
    
    def __iter__(self) -> 'BatchIterator':
        """
        Return the iterator object itself.
        
        Returns:
            BatchIterator: The iterator object
        """
        self.current_index = 0
        return self
    
    def __next__(self) -> List:
        """
        Get the next batch of data.
        
        Returns:
            List: Batch of data elements
            
        Raises:
            StopIteration: When there are no more batches
        """
        if self.current_index >= len(self.data):
            raise StopIteration
        
        batch = self.data[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        
        return batch


def load_preprocessed_data(data_dir: str) -> MSMarcoDataset:
    """
    Load preprocessed MS MARCO data and return a dataset handler.
    
    Args:
        data_dir (str): Directory containing preprocessed data
        
    Returns:
        MSMarcoDataset: Handler for the MS MARCO dataset
    """
    return MSMarcoDataset(data_dir)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="MS MARCO Dataset Handler")
    parser.add_argument("--data_dir", type=str, default="./processed_data",
                        help="Directory containing preprocessed MS MARCO data")
    
    args = parser.parse_args()
    
    # Load the dataset
    dataset = load_preprocessed_data(args.data_dir)
    
    # Print some dataset statistics
    print("\nDataset Statistics:")
    print(f"Number of passages: {len(dataset.collection)}")
    print(f"Number of training queries: {len(dataset.train_queries)}")
    print(f"Number of validation queries: {len(dataset.val_queries)}")
    print(f"Number of training relevance judgments: {len(dataset.train_qrels)}")
    print(f"Number of validation relevance judgments: {len(dataset.val_qrels)}")
    
    # Example: Get first 5 passages
    print("\nFirst 5 passages:")
    passage_iterator = dataset.get_passage_batch_iterator(batch_size=5)
    first_batch = next(iter(passage_iterator))
    for pid, passage in first_batch:
        print(f"Passage {pid}: {passage[:100]}...")
    
    # Example: Get first 3 training queries and their relevant passages
    print("\nFirst 3 training queries and their relevant passages:")
    query_iterator = dataset.get_train_query_batch_iterator(batch_size=3)
    first_batch = next(iter(query_iterator))
    for qid, query in first_batch:
        relevant_pids = dataset.get_relevant_passages(qid, split='train')
        print(f"Query {qid}: {query}")
        print(f"  Relevant passages: {len(relevant_pids)}")
        if relevant_pids:
            sample_pid = next(iter(relevant_pids))
            sample_passage = dataset.get_passage_text(sample_pid)
            print(f"  Sample passage {sample_pid}: {sample_passage[:100]}...")