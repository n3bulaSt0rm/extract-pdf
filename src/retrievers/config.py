"""
Qdrant Retriever Configuration
"""

DEFAULT_CONFIG = {
    "host": "localhost",
    "port": 6333,
    "collection_name": "hust_documents",
    "top_k": 5,
    "similarity_threshold": 0.6,
    "use_reranking": True,
    "timeout": 10.0
} 