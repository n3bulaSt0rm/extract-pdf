"""
Qdrant-based Document Retriever for RAG Pipeline

This module provides a Qdrant-based document retriever for the RAG pipeline.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class QdrantRetriever:
    """Retriever that gets documents from Qdrant."""
    
    def __init__(self, 
                client,
                collection_name: str,
                top_k: int = 5,
                similarity_threshold: float = 0.6,
                use_reranking: bool = True):
        """
        Initialize the Qdrant retriever.
        
        Args:
            client: Qdrant client
            collection_name (str): Name of the collection
            top_k (int): Number of documents to retrieve
            similarity_threshold (float): Minimum similarity score
            use_reranking (bool): Whether to use reranking
        """
        self.client = client
        self.collection_name = collection_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.use_reranking = use_reranking
        
        # Initialize embedding model for queries
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("vinai/phobert-base")
            logger.info("Initialized embedding model for queries")
        except ImportError:
            logger.error("Failed to import SentenceTransformer. Make sure it's installed.")
            raise
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve documents from Qdrant.
        
        Args:
            query (str): Query text
            
        Returns:
            List[Dict]: Retrieved documents
        """
        try:
            # Preprocess query if needed (for Vietnamese)
            try:
                from pyvi import ViTokenizer
                tokenized_query = ViTokenizer.tokenize(query)
                logger.debug(f"Tokenized query: {tokenized_query}")
            except ImportError:
                logger.warning("PyVi not available, using raw query")
                tokenized_query = query
            
            # Embed query
            query_vector = self.model.encode(tokenized_query)
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=self.top_k,
                score_threshold=self.similarity_threshold
            )
            
            logger.info(f"Retrieved {len(search_results)} documents from Qdrant")
            
            # Apply reranking if enabled
            if self.use_reranking and len(search_results) > 1:
                search_results = self._rerank_results(search_results, query)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def _rerank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Rerank results using additional criteria.
        
        Args:
            results (List[Dict]): Initial search results
            query (str): Original query
            
        Returns:
            List[Dict]: Reranked results
        """
        # This is a simple implementation - you could use a more sophisticated reranker
        try:
            # Extract query keywords
            keywords = set(query.lower().split())
            
            # Score each result based on keyword presence
            for result in results:
                content = ""
                
                # Handle both direct content and payload formats
                if "payload" in result:
                    if "content" in result["payload"]:
                        content = result["payload"]["content"]
                    elif "text" in result["payload"]:
                        content = result["payload"]["text"]
                    elif "text_preview" in result["payload"]:
                        content = result["payload"]["text_preview"]
                elif "content" in result:
                    content = result["content"]
                
                content = content.lower()
                
                # Count keyword occurrences
                keyword_count = sum(1 for keyword in keywords if keyword in content)
                keyword_score = keyword_count / max(len(keywords), 1)
                
                # Adjust score (70% vector similarity, 30% keyword match)
                result["original_score"] = result["score"]
                result["keyword_score"] = keyword_score
                result["score"] = result["score"] * 0.7 + keyword_score * 0.3
            
            # Resort by combined score
            results.sort(key=lambda x: x["score"], reverse=True)
            
            logger.debug("Reranked results based on keyword matching")
            
            return results
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return results  # Return original results on error 