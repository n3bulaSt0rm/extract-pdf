"""
Vietnamese Text Embedder Module

This module provides text embedding functionality specifically for Vietnamese text
using the AITeamVN/Vietnamese_Embedding model from Hugging Face.
"""

import os
import torch
from typing import List, Dict, Any, Optional, Union, Callable
import numpy as np
from tqdm import tqdm
from pathlib import Path

# LangChain imports
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Load environment variables (if needed)
from dotenv import load_dotenv
load_dotenv()

class VietnameseTextEmbedder:
    """
    A class that provides text embedding functionality for Vietnamese text
    using the AITeamVN/Vietnamese_Embedding model.
    """
    
    def __init__(
        self,
        model_name: str = "AITeamVN/Vietnamese_Embedding",
        max_seq_length: int = 2048,
        cache_folder: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        use_gpu: Optional[bool] = None
    ):
        """
        Initialize the Vietnamese text embedder.
        
        Args:
            model_name (str): The name of the Hugging Face embedding model
            max_seq_length (int): Maximum sequence length for the model
            cache_folder (Optional[str]): Directory to cache the model
            model_kwargs (Optional[Dict[str, Any]]): Additional kwargs for the model
            use_gpu (Optional[bool]): Whether to use GPU acceleration (None for auto-detect)
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        
        # Check for CUDA availability
        self.device = self._get_device(use_gpu)
        print(f"Using device: {self.device}")
        
        # Initialize embedding model kwargs
        if model_kwargs is None:
            model_kwargs = {"device": self.device}
        else:
            model_kwargs["device"] = self.device
        
        encode_kwargs = {"normalize_embeddings": False}  # For dot product similarity
        
        # Initialize HuggingFaceEmbeddings from langchain
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder=cache_folder
        )
        
        # Set max sequence length
        if hasattr(self.embeddings, "client") and hasattr(self.embeddings.client, "max_seq_length"):
            self.embeddings.client.max_seq_length = max_seq_length
    
    def _get_device(self, use_gpu: Optional[bool] = None) -> str:
        """
        Determine which device (CPU/CUDA) to use for embeddings.
        
        Args:
            use_gpu (Optional[bool]): Whether to use GPU. If None, auto-detect.
            
        Returns:
            str: Device string for PyTorch
        """
        if use_gpu is False:
            return "cpu"
        
        # Check for CUDA availability
        if torch.cuda.is_available():
            # Get CUDA version
            cuda_version = torch.version.cuda
            print(f"CUDA is available (version {cuda_version})")
            
            # Check if we have CUDA 12.1 as specified in requirements
            if cuda_version and "12.1" in cuda_version:
                print("Using CUDA 12.1 for Vietnamese embeddings")
            else:
                print(f"Warning: Using CUDA {cuda_version} (requirements specify 12.1)")
            
            return "cuda"
        else:
            print("CUDA is not available, falling back to CPU")
            return "cpu"
    
    def _clean_text(self, text: str) -> str:
        """
        Clean the input text before embedding.
        
        Args:
            text (str): The input text to clean
            
        Returns:
            str: The cleaned text
        """
        # Remove excessive whitespace
        text = ' '.join(text.split())
        return text
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            List[float]: The embedding vector
        """
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Generate embedding
        embedding = self.embeddings.embed_query(cleaned_text)
        return embedding
    
    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Generate embeddings for a list of LangChain Documents.
        
        Args:
            documents (List[Document]): List of Documents to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        # Extract text from documents
        texts = [doc.page_content for doc in documents]
        
        # Clean texts
        cleaned_texts = [self._clean_text(text) for text in texts]
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(cleaned_texts)
        return embeddings
    
    def embed_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> List[List[float]]:
        """
        Generate embeddings for a list of texts in batches.
        
        Args:
            texts (List[str]): List of texts to embed
            batch_size (int): Batch size for embedding
            show_progress (bool): Whether to show a progress bar
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        # Clean texts
        cleaned_texts = [self._clean_text(text) for text in texts]
        
        all_embeddings = []
        
        # Process in batches
        iterator = range(0, len(cleaned_texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings", unit="batch")
        
        for i in iterator:
            batch_texts = cleaned_texts[i:i+batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute the similarity between two embeddings using dot product.
        
        Args:
            embedding1 (List[float]): First embedding vector
            embedding2 (List[float]): Second embedding vector
            
        Returns:
            float: Similarity score between the embeddings
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Compute dot product
        return float(np.dot(vec1, vec2))
    
    def compute_similarities(self, query_embedding: List[float], document_embeddings: List[List[float]]) -> List[float]:
        """
        Compute similarities between a query embedding and multiple document embeddings.
        
        Args:
            query_embedding (List[float]): The query embedding
            document_embeddings (List[List[float]]): List of document embeddings
            
        Returns:
            List[float]: List of similarity scores
        """
        # Convert to numpy arrays
        query_vec = np.array(query_embedding)
        doc_vecs = np.array(document_embeddings)
        
        # Compute dot products
        similarities = np.dot(doc_vecs, query_vec)
        return similarities.tolist()
    
    def get_langchain_embeddings(self) -> Embeddings:
        """
        Get the underlying LangChain embeddings model.
        
        Returns:
            Embeddings: The LangChain embeddings model
        """
        return self.embeddings


class VietnameseEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings class for Vietnamese.
    This class implements the LangChain Embeddings interface.
    """
    
    def __init__(
        self,
        model_name: str = "AITeamVN/Vietnamese_Embedding",
        max_seq_length: int = 2048,
        cache_folder: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        use_gpu: Optional[bool] = None
    ):
        """
        Initialize the Vietnamese embeddings model.
        
        Args:
            model_name (str): The name of the Hugging Face embedding model
            max_seq_length (int): Maximum sequence length for the model
            cache_folder (Optional[str]): Directory to cache the model
            model_kwargs (Optional[Dict[str, Any]]): Additional kwargs for the model
            use_gpu (Optional[bool]): Whether to use GPU acceleration (None for auto-detect)
        """
        self.embedder = VietnameseTextEmbedder(
            model_name=model_name,
            max_seq_length=max_seq_length,
            cache_folder=cache_folder,
            model_kwargs=model_kwargs,
            use_gpu=use_gpu
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents using the Vietnamese embeddings model.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embeddings
        """
        return self.embedder.embed_batch(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed query text using the Vietnamese embeddings model.
        
        Args:
            text (str): Query text to embed
            
        Returns:
            List[float]: Embedding for the query
        """
        return self.embedder.embed_text(text)


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Create embedder
    embedder = VietnameseTextEmbedder()
    
    # Example texts
    queries = ["Trí tuệ nhân tạo là gì", "Lợi ích của giấc ngủ"]
    documents = [
        "Trí tuệ nhân tạo là công nghệ giúp máy móc suy nghĩ và học hỏi như con người. Nó hoạt động bằng cách thu thập dữ liệu, nhận diện mẫu và đưa ra quyết định.",
        "Giấc ngủ giúp cơ thể và não bộ nghỉ ngơi, hồi phục năng lượng và cải thiện trí nhớ. Ngủ đủ giấc giúp tinh thần tỉnh táo và làm việc hiệu quả hơn."
    ]
    
    # Generate embeddings
    query_embeddings = embedder.embed_batch(queries)
    doc_embeddings = embedder.embed_batch(documents)
    
    # Compute similarities
    for i, query_embedding in enumerate(query_embeddings):
        print(f"\nQuery: {queries[i]}")
        similarities = embedder.compute_similarities(query_embedding, doc_embeddings)
        for j, similarity in enumerate(similarities):
            print(f"  Document {j+1}: {similarity:.6f} - {documents[j][:50]}...")
    
    # Example with LangChain Embeddings interface
    lc_embeddings = VietnameseEmbeddings()
    lc_query_embeddings = lc_embeddings.embed_query(queries[0])
    print(f"\nLangChain Embeddings - dimension: {len(lc_query_embeddings)}")

    # Print GPU info if available
    if torch.cuda.is_available():
        print(f"\nGPU Information:")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
