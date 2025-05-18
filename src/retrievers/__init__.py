"""
Document Retrievers for RAG Pipeline

This package contains retriever implementations for the RAG pipeline.
"""

from src.retrievers.qdrant_retriever import QdrantRetriever
from src.retrievers.config import DEFAULT_CONFIG

def create_qdrant_retriever(config=None):
    """
    Create a Qdrant retriever with the given configuration.
    If no config is provided, use the default configuration.
    
    Args:
        config (dict, optional): Configuration dictionary
        
    Returns:
        QdrantRetriever: Initialized Qdrant retriever
    """
    if config is None:
        config = DEFAULT_CONFIG
    else:
        # Merge with defaults for any missing keys
        merged_config = DEFAULT_CONFIG.copy()
        merged_config.update(config)
        config = merged_config
    
    # Import here to avoid circular dependencies
    from qdrant_client import QdrantClient
    
    # Create Qdrant client
    client = QdrantClient(
        host=config["host"],
        port=config["port"],
        timeout=config.get("timeout", 10.0)
    )
    
    # Create retriever
    return QdrantRetriever(
        client=client,
        collection_name=config["collection_name"],
        top_k=config["top_k"],
        similarity_threshold=config["similarity_threshold"],
        use_reranking=config["use_reranking"]
    )

__all__ = ["QdrantRetriever", "create_qdrant_retriever", "DEFAULT_CONFIG"] 