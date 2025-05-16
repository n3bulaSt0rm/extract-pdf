#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

try:
    from pyvi import ViTokenizer
except ImportError:
    print("Error: Required package 'pyvi' is not installed. Please run 'pip install pyvi==0.1.1'")
    sys.exit(1)

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
except ImportError as e:
    print(f"Error: {e}")
    print("\nDependency error. Please install required packages:")
    print("pip install pyvi==0.1.1 sentence-transformers==2.2.2 huggingface-hub==0.12.1 ")
    print("pip install qdrant-client==1.4.0 torch transformers protobuf==5.26.1")
    sys.exit(1)

def tokenize_vietnamese_text(text: str) -> str:
    """
    Apply Vietnamese tokenization using ViTokenizer.
    
    Args:
        text (str): Input Vietnamese text
        
    Returns:
        str: Tokenized Vietnamese text with word boundaries marked by underscore
    """
    tokenized_text = ViTokenizer.tokenize(text)
    return tokenized_text

def remove_page_indicators_and_toc(text: str) -> str:
    """
    Remove:
    1. Page indicators like "Trang X" and numbers following them
    2. Lines containing ellipses (...)
    3. Table of contents sections (consecutive lines with ellipses)
    
    Args:
        text (str): Input text with potential page indicators and TOC
        
    Returns:
        str: Cleaned text with unwanted elements removed
    """
    # Split into lines for processing
    lines = text.splitlines()
    cleaned_lines = []
    
    i = 0
    while i < len(lines):
        current_line = lines[i].strip()
        
        # Check if this line contains a page indicator
        page_indicator_match = re.match(r'^Trang\s+\d+\s*$', current_line)
        
        # Check if this line contains ellipses
        has_ellipses = "..." in current_line or ".." in current_line
        
        # Skip TOC-like lines and page numbers
        if page_indicator_match:
            # Skip this line
            i += 1
            
            # If the next line has just a number, skip that too
            if i < len(lines) and re.match(r'^\d+\s*$', lines[i].strip()):
                i += 1
        elif has_ellipses or re.match(r'^[.\s]*$', current_line):
            # Skip lines with ellipses or just dots
            i += 1
        elif re.match(r'^(CHƯƠNG|Chương|Mục)\s+[IVXivx0-9]+', current_line):
            # Skip chapter/section headers that typically appear in TOC
            i += 1
        else:
            # Keep this line
            cleaned_lines.append(lines[i])
            i += 1
    
    return '\n'.join(cleaned_lines)

def chunk_by_article(text: str) -> List[Dict[str, Any]]:
    """
    Split text into chunks where each "Điều" (Article) is a separate chunk.
    Also captures content before the first article as a separate chunk.
    
    Args:
        text (str): The input text to be chunked
        
    Returns:
        List[Dict[str, Any]]: List of chunks with metadata
    """
    # Remove page indicators, TOC, and other unwanted elements first
    text = remove_page_indicators_and_toc(text)
    
    # Split text into lines
    lines = text.splitlines()
    
    # Further clean consecutive empty lines
    cleaned_lines = []
    prev_empty = False
    for line in lines:
        is_empty = not line.strip()
        if is_empty and prev_empty:
            continue  # Skip consecutive empty lines
        cleaned_lines.append(line)
        prev_empty = is_empty
    
    lines = cleaned_lines
    
    chunks = []
    current_chunk = []
    current_article = ""
    start_line = 1  # Start from line 1 for human readability
    current_line = 0
    found_first_article = False
    
    for i, line in enumerate(lines):
        current_line = i + 1  # 1-based line numbering
        
        # Check if this line starts a new article (Điều)
        article_match = re.match(r'^Điều\s+(\d+)[.:]\s+(.*)', line)
        
        if article_match:
            # If we've found the first article and have content before it
            if not found_first_article and current_chunk:
                # Save the content before the first article as a separate chunk
                chunk_text = "\n".join(current_chunk)
                chunk_text = tokenize_vietnamese_text(chunk_text)
                
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        "article": "Phần mở đầu",  # Introduction/Preamble
                        "start_line": start_line,
                        "end_line": current_line - 1
                    }
                })
                found_first_article = True
            
            # If we've accumulated content for a previous article, save that chunk
            elif found_first_article and current_chunk:
                chunk_text = "\n".join(current_chunk)
                chunk_text = tokenize_vietnamese_text(chunk_text)
                
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        "article": current_article,
                        "start_line": start_line,
                        "end_line": current_line - 1
                    }
                })
            
            # Start a new chunk
            found_first_article = True
            current_article = f"Điều {article_match.group(1)}: {article_match.group(2)}"
            current_chunk = [line]
            start_line = current_line
        else:
            # Continue with the current chunk
            current_chunk.append(line)
    
    # Don't forget to add the last chunk if it exists
    if current_chunk:
        chunk_text = "\n".join(current_chunk)
        chunk_text = tokenize_vietnamese_text(chunk_text)
        
        chunks.append({
            "content": chunk_text,
            "metadata": {
                "article": current_article if found_first_article else "Phần mở đầu",
                "start_line": start_line,
                "end_line": current_line
            }
        })
    
    return chunks

def verify_chunking(chunks: List[Dict], total_lines: int) -> bool:
    """
    Verify that all lines from the original text are accounted for in the chunks.
    
    Args:
        chunks (List[Dict]): The chunks to verify
        total_lines (int): Total number of lines in the original text
        
    Returns:
        bool: True if all lines are accounted for, False otherwise
    """
    # Check if we have any chunks at all
    if not chunks:
        return False
    
    # Check that the first chunk starts at line 1
    if chunks[0]["metadata"]["start_line"] != 1:
        return False
    
    # Check that the last chunk ends at the final line
    if chunks[-1]["metadata"]["end_line"] != total_lines:
        return False
    
    # Check for gaps or overlaps in line numbering
    for i in range(len(chunks) - 1):
        current_end = chunks[i]["metadata"]["end_line"]
        next_start = chunks[i+1]["metadata"]["start_line"]
        
        # There should be no gaps or overlaps
        if next_start != current_end + 1:
            return False
    
    return True

def encode_chunks(chunks: List[Dict], model_name: str = "vinai/phobert-base") -> List[np.ndarray]:
    """
    Encode chunks using a Vietnamese language model.
    
    Args:
        chunks (List[Dict]): List of text chunks
        model_name (str): Name of the sentence transformer model to use
        
    Returns:
        List[np.ndarray]: List of embeddings for each chunk
    """
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Encoding {len(chunks)} chunks...")
    
    # Extract just the text content for encoding
    texts = [chunk["content"] for chunk in chunks]
    
    # Convert the texts to embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    
    return embeddings

def create_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: str = "Cosine"
) -> None:
    """
    Create a collection in Qdrant if it doesn't exist.
    
    Args:
        client (QdrantClient): Qdrant client
        collection_name (str): Name of the collection
        vector_size (int): Size of the vectors
        distance (str): Distance metric to use
    """
    # Check if collection already exists
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if collection_name in collection_names:
        print(f"Collection '{collection_name}' already exists")
        return
    
    # Create the collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=distance
        )
    )
    print(f"Created collection '{collection_name}'")

def store_chunks_to_qdrant(
    chunks: List[Dict],
    embeddings: List[np.ndarray],
    host: str = "localhost",
    port: int = 6333,
    collection_name: str = "hust_documents"
) -> None:
    """
    Store chunks and their embeddings in Qdrant.
    
    Args:
        chunks (List[Dict]): List of text chunks with metadata
        embeddings (List[np.ndarray]): List of embeddings for each chunk
        host (str): Qdrant server host
        port (int): Qdrant server port
        collection_name (str): Name of the collection to store data
    """
    # Connect to Qdrant
    client = QdrantClient(host=host, port=port)
    
    # Create collection if it doesn't exist
    vector_size = len(embeddings[0])
    create_collection(client, collection_name, vector_size)
    
    # Prepare points for upload
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Extract metadata
        metadata = chunk["metadata"]
        metadata["text"] = chunk["content"][:100]  # Store the first 100 chars for preview
        
        points.append(models.PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload=metadata
        ))
    
    # Upload to Qdrant
    print(f"Uploading {len(points)} points to Qdrant...")
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    print(f"Successfully stored {len(points)} chunks in Qdrant collection '{collection_name}'")

def process_file(
    file_path: str, 
    output_dir: Optional[str] = None,
    store_in_qdrant: bool = True,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    qdrant_collection: str = "hust_documents"
) -> List[Dict[str, Any]]:
    """
    Process a text file by tokenizing, chunking it by articles, and optionally storing in Qdrant.
    
    Args:
        file_path (str): Path to the text file to be processed
        output_dir (Optional[str]): Directory to save the chunks as JSON. If None, chunks will not be saved.
        store_in_qdrant (bool): Whether to store chunks in Qdrant
        qdrant_host (str): Qdrant server host
        qdrant_port (int): Qdrant server port
        qdrant_collection (str): Name of the Qdrant collection
        
    Returns:
        List[Dict[str, Any]]: List of chunks with metadata
    """
    print(f"Processing file: {file_path}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Get the total number of lines in the file
    total_lines = len(text.splitlines())
    
    # Chunk the text
    chunks = chunk_by_article(text)
    
    # Verify that all lines are accounted for
    if not verify_chunking(chunks, total_lines):
        print(f"WARNING: Line verification failed. Some content may be missing or duplicated.")
    else:
        print(f"Verification successful: All {total_lines} lines accounted for in {len(chunks)} chunks.")
    
    # Save chunks to JSON if output_dir is provided
    if output_dir:
        output_path = Path(output_dir) / f"{Path(file_path).stem}_chunks.json"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(chunks)} chunks to {output_path}")
    
    # Store in Qdrant if requested
    if store_in_qdrant:
        try:
            # Encode chunks
            embeddings = encode_chunks(chunks)
            
            # Store in Qdrant
            store_chunks_to_qdrant(
                chunks=chunks,
                embeddings=embeddings,
                host=qdrant_host,
                port=qdrant_port,
                collection_name=qdrant_collection
            )
        except Exception as e:
            print(f"Error storing chunks in Qdrant: {e}")
            print("Make sure Qdrant is running and the required dependencies are installed.")
    
    return chunks

def main():
    # Set the specific file to process
    file_path = r"D:\DATN_HUST\test\output\QCDT-2023-upload.txt"
    chunks_dir = "chunks"
    
    # Create the chunks directory if it doesn't exist
    os.makedirs(chunks_dir, exist_ok=True)
    
    print(f"Processing file: {file_path}")
    
    # Configure Qdrant
    store_in_qdrant = True
    qdrant_host = "localhost"
    qdrant_port = 6333
    qdrant_collection = "hust_documents"
    
    # Process the file
    chunks = process_file(
        file_path=file_path, 
        output_dir=chunks_dir,
        store_in_qdrant=store_in_qdrant,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        qdrant_collection=qdrant_collection
    )
    
    print(f"Successfully processed {len(chunks)} chunks from {os.path.basename(file_path)}")
    
    # Print a sample of the first chunk for verification
    if chunks:
        print("\nSample of first chunk:")
        print(f"Article: {chunks[0]['metadata']['article']}")
        print(f"Lines: {chunks[0]['metadata']['start_line']} - {chunks[0]['metadata']['end_line']}")
        print(f"Content (first 200 chars): {chunks[0]['content'][:200]}...")

if __name__ == "__main__":
    main()
