#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from pyvi import ViTokenizer

class VietnameseDocumentRetriever:
    """
    A class for retrieving Vietnamese documents from Qdrant based on semantic search.
    """
    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "hust_documents"
    ):
        """
        Initialize the document retriever with the specified model and Qdrant connection.
        
        Args:
            model_name (str): The model to use for encoding queries
            host (str): Qdrant server host
            port (int): Qdrant server port
            collection_name (str): Name of the collection to query
        """
        try:
            self.model = SentenceTransformer(model_name)
            print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise
            
        try:
            self.client = QdrantClient(host=host, port=port)
            self.collection_name = collection_name
            
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                raise ValueError(f"Collection '{self.collection_name}' does not exist in Qdrant")
            
            print(f"Connected to Qdrant collection: {collection_name}")
            
            # Get collection info
            collection_info = self.client.get_collection(collection_name=collection_name)
            num_vectors = collection_info.points_count
            print(f"Collection contains {num_vectors} documents")
            
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}")
            raise
    
    def tokenize_query(self, query: str) -> str:
        """
        Tokenize the query using ViTokenizer.
        
        Args:
            query (str): The query text
            
        Returns:
            str: Tokenized query
        """
        return ViTokenizer.tokenize(query)
    
    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: Optional[float] = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Search for documents that match the query.
        
        Args:
            query (str): The search query
            limit (int): Maximum number of results to return
            score_threshold (Optional[float]): Minimum score threshold for results
            
        Returns:
            List[Dict[str, Any]]: List of matching documents with scores
        """
        # Tokenize the query
        tokenized_query = self.tokenize_query(query)
        print(f"Tokenized query: {tokenized_query}")
        
        # Encode the query
        query_vector = self.model.encode(tokenized_query)
        
        # Search in Qdrant
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=limit,
            score_threshold=score_threshold
        )
        
        # Format the results
        results = []
        for hit in search_result:
            results.append({
                "score": hit.score,
                "article": hit.payload.get("article", ""),
                "start_line": hit.payload.get("start_line", 0),
                "end_line": hit.payload.get("end_line", 0),
                "preview": hit.payload.get("text", ""),
                "id": hit.id
            })
        
        return results

def display_results(results: List[Dict[str, Any]]) -> None:
    """
    Display the search results in a readable format.
    
    Args:
        results (List[Dict[str, Any]]): The search results to display
    """
    if not results:
        print("No results found.")
        return
    
    print(f"\nFound {len(results)} results:\n")
    
    for i, result in enumerate(results):
        print(f"Result {i+1} [Score: {result['score']:.4f}] [ID: {result['id']}]")
        print(f"Article: {result['article']}")
        print(f"Lines: {result['start_line']} - {result['end_line']} ({result['end_line'] - result['start_line'] + 1} lines)")
        print(f"Preview: {result['preview']}...")
        print("-" * 80)

def main():
    # Initialize the retriever
    try:
        retriever = VietnameseDocumentRetriever()
    except Exception as e:
        print(f"Error initializing document retriever: {e}")
        print("Make sure Qdrant is running and you've stored documents using store_to_qdrant.py")
        return
    
    # Sample queries to demonstrate different types of searches
    sample_queries = [
        "Chương trình đào tạo là gì?",
        "Thời gian đào tạo tiến sĩ",
        "Định nghĩa tín chỉ",
        "Quy định về học phần",
        "Điều kiện tốt nghiệp",
        "Quy định về thời gian học tập"
    ]
    
    # Print sample queries
    print("\nSample queries:")
    for i, query in enumerate(sample_queries):
        print(f"{i+1}. {query}")
    
    # Allow user to select a query or enter a custom one
    print("\nEnter a number to select a sample query, or type your own query:")
    user_input = input("> ")
    
    if user_input.isdigit() and 1 <= int(user_input) <= len(sample_queries):
        query = sample_queries[int(user_input) - 1]
    else:
        query = user_input
    
    print(f"\nSearching for: {query}")
    
    try:
        results = retriever.search(query)
        display_results(results)
    except Exception as e:
        print(f"Error during search: {e}")

if __name__ == "__main__":
    main() 