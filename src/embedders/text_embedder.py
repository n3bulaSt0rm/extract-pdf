#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union

try:
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
except ImportError as e:
    print(f"Error: {e}")
    print("\nDependency error. Please install required packages:")
    print("pip install sentence-transformers==2.2.2 huggingface-hub==0.12.1 ")
    print("pip install qdrant-client==1.4.0 torch transformers protobuf==5.26.1")
    sys.exit(1)

class TextEmbedder:
    """
    A class for embedding texts and storing them in a vector database.
    """
    
    def __init__(self, 
                model_name: str = "vinai/phobert-base",
                batch_size: int = 8):
        """
        Initialize the text embedder.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
            batch_size (int): Batch size for embedding
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._load_model()
        
    def _load_model(self):
        """Load the embedding model."""
        print(f"Loading model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def encode_chunks(self, chunks: List[Dict]) -> List[np.ndarray]:
        """
        Encode chunks using the Vietnamese language model.
        
        Args:
            chunks (List[Dict]): List of text chunks
            
        Returns:
            List[np.ndarray]: List of embeddings for each chunk
        """
        print(f"Encoding {len(chunks)} chunks...")
        
        # Extract just the text content for encoding
        texts = [chunk["content"] for chunk in chunks]
        
        # Convert the texts to embeddings in smaller batches to avoid memory issues
        batch_size = self.batch_size
        embeddings = []
        
        total_batches = (len(texts) + batch_size - 1) // batch_size  # Ceiling division
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{total_batches} ({len(batch_texts)} chunks)")
            
            try:
                # Encode batch
                batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
                embeddings.extend(batch_embeddings)
                print(f"✓ Batch encoded successfully")
            except Exception as e:
                print(f"✗ Error in batch: {e}")
                
                # Process one by one if batch fails
                print("Trying to encode chunks individually...")
                for j, text in enumerate(batch_texts):
                    try:
                        # Encode single text
                        emb = self.model.encode([text], show_progress_bar=False)[0]
                        embeddings.append(emb)
                        print(f"  ✓ Chunk {i+j+1}/{len(texts)} encoded successfully")
                    except Exception as e2:
                        print(f"  ✗ Failed to encode chunk {i+j+1}: {e2}")
                        
                        # Create placeholder embedding
                        if embeddings:
                            dummy_vector = np.zeros_like(embeddings[0])
                        else:
                            dummy_vector = np.zeros(768)  # Default size for PhoBERT
                        
                        embeddings.append(dummy_vector)
                        print(f"  ! Added zero vector as placeholder")
        
        # Verify counts
        if len(embeddings) != len(chunks):
            print(f"Warning: Embeddings count ({len(embeddings)}) doesn't match chunks count ({len(chunks)})")
            if len(embeddings) > len(chunks):
                embeddings = embeddings[:len(chunks)]
            else:
                # Add dummy vectors if needed
                while len(embeddings) < len(chunks):
                    if embeddings:
                        dummy_vector = np.zeros_like(embeddings[0])
                    else:
                        dummy_vector = np.zeros(768)
                    embeddings.append(dummy_vector)
        
        return embeddings

    def encode_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode individual texts using the embedding model.
        
        Args:
            texts (List[str]): List of texts to encode
            
        Returns:
            List[np.ndarray]: List of embeddings
        """
        if not texts:
            return []
        
        print(f"Encoding {len(texts)} texts...")
        
        # Convert the texts to embeddings in smaller batches to avoid memory issues
        batch_size = self.batch_size
        embeddings = []
        
        total_batches = (len(texts) + batch_size - 1) // batch_size  # Ceiling division
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{total_batches} ({len(batch_texts)} texts)")
            
            try:
                # Encode batch
                batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
                embeddings.extend(batch_embeddings)
                print(f"✓ Batch encoded successfully")
            except Exception as e:
                print(f"✗ Error in batch: {e}")
                
                # Process one by one if batch fails
                for j, text in enumerate(batch_texts):
                    try:
                        emb = self.model.encode([text], show_progress_bar=False)[0]
                        embeddings.append(emb)
                    except Exception as e2:
                        print(f"  ✗ Failed to encode text: {e2}")
                        # Create placeholder embedding
                        if embeddings:
                            dummy_vector = np.zeros_like(embeddings[0])
                        else:
                            dummy_vector = np.zeros(768)  # Default size for PhoBERT
                        embeddings.append(dummy_vector)
        
        return embeddings

    def save_embeddings(self, chunks: List[Dict], embeddings: List[np.ndarray], 
                       output_path: Optional[str] = None) -> None:
        """
        Save chunks and their embeddings to a file.
        
        Args:
            chunks (List[Dict]): List of text chunks with metadata
            embeddings (List[np.ndarray]): List of embeddings for each chunk
            output_path (str, optional): Path to save the embeddings
        """
        if output_path:
            print(f"Saving chunks and embeddings to {output_path}")
            
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Prepare data for saving
            serializable_embeddings = [embedding.tolist() for embedding in embeddings]
            data = {
                "chunks": chunks,
                "embeddings": serializable_embeddings,
                "model_name": self.model_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"✓ Saved {len(chunks)} chunks and embeddings")


class QdrantStorage:
    """
    A class for storing text chunks and their embeddings in Qdrant.
    """
    
    def __init__(self, 
                host: str = "localhost",
                port: int = 6333,
                collection_name: str = "hust_documents",
                distance: str = "Cosine"):
        """
        Initialize the Qdrant storage.
        
        Args:
            host (str): Qdrant server host
            port (int): Qdrant server port
            collection_name (str): Name of the collection to store data
            distance (str): Distance metric to use ("Cosine", "Euclid", "Dot")
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.distance = distance
        self._connect_to_qdrant()
    
    def _connect_to_qdrant(self):
        """Connect to Qdrant server."""
        try:
            # Connect to Qdrant
            print(f"Connecting to Qdrant at {self.host}:{self.port}...")
            self.client = QdrantClient(host=self.host, port=self.port)
            print("Connected successfully")
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}")
            print("Make sure Qdrant is running and accessible")
            raise
    
    def create_collection(self, vector_size: int = 768) -> bool:
        """
        Create a collection in Qdrant if it doesn't exist.
        
        Args:
            vector_size (int): Size of the vectors
            
        Returns:
            bool: True if successful
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name in collection_names:
                print(f"Collection '{self.collection_name}' already exists")
                return True
            
            # Create the collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=self.distance
                )
            )
            print(f"Created collection '{self.collection_name}'")
            return True
        except Exception as e:
            print(f"Error creating collection: {e}")
            return False
    
    def store_chunks(self, chunks: List[Dict], embeddings: List[np.ndarray]) -> bool:
        """
        Store chunks and their embeddings in Qdrant.
        
        Args:
            chunks (List[Dict]): List of text chunks with metadata
            embeddings (List[np.ndarray]): List of embeddings for each chunk
            
        Returns:
            bool: True if successful
        """
        # Check that chunks and embeddings match
        if len(chunks) != len(embeddings):
            print(f"Error: Number of chunks ({len(chunks)}) does not match number of embeddings ({len(embeddings)})")
            return False
        
        try:
            # Create collection if it doesn't exist
            vector_size = len(embeddings[0])
            self.create_collection(vector_size)
            
            # Upload in batches to avoid memory issues
            batch_size = 20
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            total_uploaded = 0
            
            for i in range(0, len(chunks), batch_size):
                end_idx = min(i + batch_size, len(chunks))
                batch_chunks = chunks[i:end_idx]
                batch_embeddings = embeddings[i:end_idx]
                
                print(f"\nUploading batch {i//batch_size + 1}/{total_batches} ({len(batch_chunks)} chunks)")
                
                # Prepare points for upload
                points = []
                for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                    try:
                        # Extract metadata and create preview
                        metadata = chunk["metadata"]
                        text_preview = chunk["content"][:100] if chunk["content"] else ""  # First 100 chars
                        
                        # Create simplified payload (avoid nested structures)
                        payload = {
                            "article": str(metadata.get("article", "")),
                            "article_idx": int(metadata.get("article_idx", 0)),
                            "chunk_idx": int(metadata.get("chunk_idx", 0)),
                            "total_chunks": int(metadata.get("total_chunks", 1)),
                            "start_line": int(metadata.get("start_line", 0)),
                            "end_line": int(metadata.get("end_line", 0)),
                            "is_complete_article": bool(metadata.get("is_complete_article", False)),
                            "sentence_range": str(metadata.get("sentence_range", "")),
                            "sentences": int(metadata.get("sentences", 0)),
                            "total_sentences": int(metadata.get("total_sentences", 0)) if "total_sentences" in metadata else 0,
                            "text_preview": text_preview,
                            "content": chunk["content"]
                        }
                        
                        # Create point
                        point_id = i + j
                        vector = embedding.tolist()
                        
                        # Validate vector
                        if not all(isinstance(x, (int, float)) for x in vector):
                            print(f"  Warning: Invalid values in vector {point_id}. Skipping.")
                            continue
                        
                        points.append(models.PointStruct(
                            id=point_id,
                            vector=vector,
                            payload=payload
                        ))
                        
                    except Exception as e:
                        print(f"  Error creating point {i+j}: {e}")
                
                # Upload batch
                if points:
                    try:
                        self.client.upsert(
                            collection_name=self.collection_name,
                            points=points
                        )
                        total_uploaded += len(points)
                        print(f"✓ Successfully uploaded {len(points)} points")
                    except Exception as e:
                        print(f"✗ Error uploading batch: {e}")
                        print(f"  Error details: {str(e)}")
                else:
                    print("! No valid points in this batch")
            
            print(f"\nCompleted uploading {total_uploaded}/{len(chunks)} chunks to Qdrant")
            return total_uploaded > 0
        
        except Exception as e:
            print(f"Error storing chunks in Qdrant: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def is_running(self) -> bool:
        """
        Check if Qdrant server is accessible.
        
        Returns:
            bool: True if Qdrant is accessible
        """
        try:
            # Try to get collections info
            self.client.get_collections()
            return True
        except Exception as e:
            print(f"Qdrant server not accessible at {self.host}:{self.port}: {e}")
            return False


class TextProcessor:
    """
    A class that combines chunking, embedding, and storage.
    """
    
    def __init__(self,
                embedder: TextEmbedder,
                storage: QdrantStorage,
                output_dir: Optional[str] = None):
        """
        Initialize the text processor.
        
        Args:
            embedder (TextEmbedder): The embedder to use
            storage (QdrantStorage): The storage to use
            output_dir (str, optional): Directory to save embeddings
        """
        self.embedder = embedder
        self.storage = storage
        self.output_dir = output_dir
        
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
    
    def process_chunks(self, chunks: List[Dict], 
                      save_to_file: bool = True,
                      store_in_qdrant: bool = True) -> Tuple[List[Dict], List[np.ndarray]]:
        """
        Process text chunks: embed and store.
        
        Args:
            chunks (List[Dict]): List of text chunks
            save_to_file (bool): Whether to save embeddings to file
            store_in_qdrant (bool): Whether to store in Qdrant
            
        Returns:
            Tuple[List[Dict], List[np.ndarray]]: Processed chunks and their embeddings
        """
        # Embed chunks
        start_time = time.time()
        embeddings = self.embedder.encode_chunks(chunks)
        embedding_time = time.time() - start_time
        print(f"Embedding completed in {embedding_time:.2f} seconds")
        
        # Save to file if requested
        if save_to_file and self.output_dir:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"embeddings_{timestamp}.json")
            self.embedder.save_embeddings(chunks, embeddings, output_path)
        
        # Store in Qdrant if requested
        if store_in_qdrant:
            if self.storage.is_running():
                start_time = time.time()
                success = self.storage.store_chunks(chunks, embeddings)
                storage_time = time.time() - start_time
                print(f"Storage {'completed' if success else 'failed'} in {storage_time:.2f} seconds")
            else:
                print("Skipping Qdrant storage as server is not accessible")
        
        return chunks, embeddings


def main():
    """Example usage of text embedding and storage."""
    # Example chunks (these would normally come from a chunker)
    chunks = [
        {
            "content": "Đây là một ví dụ về chunk văn bản tiếng Việt đầu tiên.",
            "metadata": {
                "article": "Điều 1: Quy định chung",
                "article_idx": 0,
                "chunk_idx": 0,
                "total_chunks": 2,
                "start_line": 1,
                "end_line": 5,
                "is_complete_article": False,
                "sentence_range": "0-1",
                "sentences": 2
            }
        },
        {
            "content": "Đây là ví dụ thứ hai về chunk văn bản tiếng Việt.",
            "metadata": {
                "article": "Điều 1: Quy định chung",
                "article_idx": 0,
                "chunk_idx": 1,
                "total_chunks": 2,
                "start_line": 6,
                "end_line": 10,
                "is_complete_article": False,
                "sentence_range": "2-3",
                "sentences": 2
            }
        }
    ]
    
    # Create embedder and storage
    try:
        embedder = TextEmbedder(model_name="vinai/phobert-base")
        storage = QdrantStorage(
            host="localhost",
            port=6333,
            collection_name="test_documents"
        )
        
        # Create processor
        processor = TextProcessor(
            embedder=embedder,
            storage=storage,
            output_dir="embeddings"
        )
        
        # Process chunks
        processed_chunks, embeddings = processor.process_chunks(
            chunks=chunks,
            save_to_file=True,
            store_in_qdrant=True
        )
        
        print(f"Successfully processed {len(processed_chunks)} chunks")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 