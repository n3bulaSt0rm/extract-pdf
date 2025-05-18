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
        collection_name: str = "hust_documents",
        qdrant_version: str = "1.4.0"
    ):
        """
        Initialize the document retriever with the specified model and Qdrant connection.
        
        Args:
            model_name (str): The model to use for encoding queries
            host (str): Qdrant server host
            port (int): Qdrant server port
            collection_name (str): Name of the collection to query
            qdrant_version (str): Version of Qdrant client to use for compatibility
        """
        try:
            self.model = SentenceTransformer(model_name)
            print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise
            
        try:
            self.client = QdrantClient(host=host, port=port, timeout=10.0)
            self.collection_name = collection_name
            
            # Check if collection exists - with error handling for API differences
            try:
                collections = self.client.get_collections().collections
                collection_names = [collection.name for collection in collections]
                
                if self.collection_name not in collection_names:
                    raise ValueError(f"Collection '{self.collection_name}' does not exist in Qdrant")
                
                print(f"Connected to Qdrant collection: {collection_name}")
                
                # Try to get collection info with compatibility handling
                try:
                    collection_info = self.client.get_collection(collection_name=collection_name)
                    # Try different attribute paths based on version
                    if hasattr(collection_info, 'points_count'):
                        num_vectors = collection_info.points_count
                    elif hasattr(collection_info, 'result') and hasattr(collection_info.result, 'vectors_count'):
                        num_vectors = collection_info.result.vectors_count
                    else:
                        num_vectors = "unknown"  # Fallback
                    print(f"Collection contains {num_vectors} documents")
                except Exception as e:
                    print(f"Warning: Could not get detailed collection info: {e}")
                    print("Continuing with basic functionality...")
                
            except Exception as e:
                print(f"Warning: Error checking collections using API: {e}")
                print("Will attempt search operations directly...")
            
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
    
    def _get_all_article_chunks(self, article_idx: int) -> List[Dict[str, Any]]:
        """
        Get all chunks that belong to the same article.
        
        Args:
            article_idx (int): The article index to find chunks for
            
        Returns:
            List[Dict[str, Any]]: List of all chunks from the article
        """
        print(f"Retrieving all chunks for article_idx={article_idx}")
        
        try:
            # Create filter for the article_idx
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="article_idx",
                        match=models.MatchValue(value=article_idx)
                    )
                ]
            )
            
            # Try using scroll API first
            try:
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_condition,
                    limit=100  # Get up to 100 chunks per article
                )
                points = result[0]
                print(f"Found {len(points)} chunks using scroll API")
                
                # Extract relevant information from points
                chunks = []
                for point in points:
                    chunk_data = {
                        "id": point.id,
                        "article": point.payload.get("article", ""),
                        "start_line": point.payload.get("start_line", 0),
                        "end_line": point.payload.get("end_line", 0),
                        "preview": point.payload.get("text_preview", ""),
                        "chunk_idx": point.payload.get("chunk_idx", 0),
                        "word_range": point.payload.get("word_range", "")
                    }
                    chunks.append(chunk_data)
                
                # Sort chunks by chunk_idx
                chunks.sort(key=lambda x: x.get("chunk_idx", 0))
                return chunks
                
            except Exception as scroll_error:
                print(f"Scroll API error: {scroll_error}")
                print("Falling back to search API...")
                
                # Use search API as fallback
                dummy_vector = np.zeros(768)  # Default size for PhoBERT
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=dummy_vector.tolist(),
                    query_filter=filter_condition,
                    limit=100,  # Get up to 100 chunks per article
                    score_threshold=None  # No score threshold since we're using a dummy vector
                )
                
                print(f"Found {len(search_result)} chunks using search API")
                
                # Extract relevant information
                chunks = []
                for hit in search_result:
                    chunk_data = {
                        "id": hit.id,
                        "article": hit.payload.get("article", ""),
                        "start_line": hit.payload.get("start_line", 0),
                        "end_line": hit.payload.get("end_line", 0),
                        "preview": hit.payload.get("text_preview", ""),
                        "chunk_idx": hit.payload.get("chunk_idx", 0),
                        "word_range": hit.payload.get("word_range", "")
                    }
                    chunks.append(chunk_data)
                
                # Sort chunks by chunk_idx
                chunks.sort(key=lambda x: x.get("chunk_idx", 0))
                return chunks
                
        except Exception as e:
            print(f"Error retrieving article chunks: {e}")
            return []
    
    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: Optional[float] = 0.6,
        include_full_articles: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for documents that match the query.
        
        Args:
            query (str): The search query
            limit (int): Maximum number of results to return
            score_threshold (Optional[float]): Minimum score threshold for results
            include_full_articles (bool): Whether to include all chunks from the same article
            
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
        
        # Process the search results
        if not include_full_articles:
            # Simple format without getting full articles
            results = []
            for hit in search_result:
                # Check for preview field (which could be either "text" or "text_preview")
                preview = hit.payload.get("text_preview", "")
                if not preview:
                    preview = hit.payload.get("text", "")  # Try alternative field name
                
                results.append({
                    "score": hit.score,
                    "article": hit.payload.get("article", ""),
                    "start_line": hit.payload.get("start_line", 0),
                    "end_line": hit.payload.get("end_line", 0),
                    "preview": preview,
                    "id": hit.id
                })
            
            return results
        else:
            # Enhanced format with full article content
            results = []
            processed_articles = set()  # Track which articles we've already processed
            
            for hit in search_result:
                article_idx = hit.payload.get("article_idx", None)
                
                # Skip if we can't determine the article index or already processed this article
                if article_idx is None or article_idx in processed_articles:
                    continue
                    
                processed_articles.add(article_idx)
                
                # Get all chunks belonging to this article
                article_chunks = self._get_all_article_chunks(article_idx)
                
                if not article_chunks:
                    # If we couldn't get article chunks, just add the original hit
                    # Check for preview field (which could be either "text" or "text_preview")
                    preview = hit.payload.get("text_preview", "")
                    if not preview:
                        preview = hit.payload.get("text", "")  # Try alternative field name
                    
                    results.append({
                        "score": hit.score,
                        "article": hit.payload.get("article", ""),
                        "start_line": hit.payload.get("start_line", 0),
                        "end_line": hit.payload.get("end_line", 0),
                        "preview": preview,
                        "id": hit.id,
                        "is_full_article": False
                    })
                else:
                    # Combine all chunks into a single result
                    article_title = article_chunks[0].get("article", "")
                    combined_preview = "\n".join([chunk.get("preview", "") for chunk in article_chunks])
                    
                    results.append({
                        "score": hit.score,
                        "article": article_title,
                        "start_line": min([chunk.get("start_line", 0) for chunk in article_chunks]),
                        "end_line": max([chunk.get("end_line", 0) for chunk in article_chunks]),
                        "preview": combined_preview,
                        "id": hit.id,
                        "chunks_count": len(article_chunks),
                        "is_full_article": True
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
        
        # Show if this is a full article
        if result.get("is_full_article", False):
            print(f"FULL ARTICLE: Yes (Contains {result.get('chunks_count', '?')} chunks)")
        
        # Show preview (truncated if too long)
        preview = result['preview']
        if len(preview) > 300:
            print(f"Preview: {preview[:300]}... (truncated, total length: {len(preview)} characters)")
        else:
            print(f"Preview: {preview}")
            
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
    
    # Explain full article retrieval feature
    print("\nFULL ARTICLE RETRIEVAL: When a chunk is found, the system will automatically")
    print("retrieve all related chunks from the same article to give you complete context.")
    
    # Ask if user wants to use full article retrieval
    print("\nDo you want to retrieve full articles? (y/n) [Default: y]")
    use_full_articles = input("> ").strip().lower() != 'n'
    
    # Allow user to select a query or enter a custom one
    print("\nEnter a number to select a sample query, or type your own query:")
    user_input = input("> ")
    
    if user_input.isdigit() and 1 <= int(user_input) <= len(sample_queries):
        query = sample_queries[int(user_input) - 1]
    else:
        query = user_input
    
    print(f"\nSearching for: {query}")
    print(f"Full article retrieval: {'Enabled' if use_full_articles else 'Disabled'}")
    
    try:
        # Pass the full article option to the search method
        results = retriever.search(query, include_full_articles=use_full_articles)
        display_results(results)
    except Exception as e:
        print(f"Error during search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 