#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import sys
import nltk
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
    print("pip install qdrant-client==1.4.0 torch transformers protobuf==5.26.1 nltk")
    sys.exit(1)

# Download NLTK punkt for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

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

def split_into_sentences(text: str) -> List[str]:
    """
    Split Vietnamese text into sentences using NLTK.
    
    Args:
        text (str): Input text to split into sentences
        
    Returns:
        List[str]: List of sentences
    """
    # Handle empty or None input
    if not text or not text.strip():
        return []
    
    try:
        # Custom sentence tokenization for Vietnamese
        # First, use NLTK's default sentence tokenizer
        sentences = nltk.sent_tokenize(text)
        
        # Further split sentences on common Vietnamese sentence boundaries
        # that might not be caught by NLTK
        result = []
        for sentence in sentences:
            # Skip empty sentences
            if not sentence.strip():
                continue
                
            # Split on Vietnamese-specific patterns if they're followed by a capital letter
            parts = re.split(r'([.!?;:]\s+)(?=[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺỀẾỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ])', sentence)
            
            current_part = ""
            for i, part in enumerate(parts):
                if re.match(r'[.!?;:]\s+', part):
                    # This is a delimiter, add it to the current part and save
                    current_part += part
                    result.append(current_part)
                    current_part = ""
                else:
                    # This is content, add it to the current part
                    current_part += part
            
            # Add any remaining content
            if current_part and current_part.strip():
                result.append(current_part)
        
        # Clean up empty sentences and ensure proper capitalization
        cleaned_sentences = []
        for sentence in result:
            sentence = sentence.strip()
            if sentence:
                if not re.match(r'^[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺỀẾỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ]', sentence):
                    # If sentence doesn't start with capital letter, it might be a continuation
                    if cleaned_sentences:
                        cleaned_sentences[-1] += " " + sentence
                        continue
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    except Exception as e:
        print(f"Error splitting sentences: {e}")
        # Fall back to simple splitting by periods if NLTK fails
        simple_sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in simple_sentences if s.strip()]

def create_overlapping_chunks(
    sentences: List[str], 
    chunk_size: int = 4,
    overlap_size: int = 2
) -> List[str]:
    """
    Create overlapping chunks from a list of sentences.
    
    Args:
        sentences (List[str]): List of sentences to chunk
        chunk_size (int): Number of sentences per chunk
        overlap_size (int): Number of sentences to overlap between chunks
        
    Returns:
        List[str]: List of text chunks with overlap
    """
    if not sentences:
        return []
    
    # Ensure valid parameters
    chunk_size = max(1, min(chunk_size, len(sentences)))
    overlap_size = max(0, min(overlap_size, chunk_size - 1))
    
    chunks = []
    
    # Create overlapping chunks
    i = 0
    while i < len(sentences):
        # Get current chunk of sentences
        end_idx = min(i + chunk_size, len(sentences))
        current_chunk = sentences[i:end_idx]
        
        # Join the sentences into text
        chunk_text = " ".join(current_chunk)
        chunks.append(chunk_text)
        
        # Move index forward, accounting for overlap
        i += max(1, chunk_size - overlap_size)
    
    return chunks

def chunk_by_article(text: str, max_chunk_sentences: int = 2, overlap_sentences: int = 1) -> List[Dict[str, Any]]:
    """
    Split text into chunks where each "Điều" (Article) is split into overlapping chunks based on sentences.
    Each chunk contains a fixed number of sentences with overlap.
    Also captures content before the first article as a separate chunk.
    
    Args:
        text (str): The input text to be chunked
        max_chunk_sentences (int): Maximum number of sentences per chunk (default: 2)
        overlap_sentences (int): Number of overlapping sentences between chunks (default: 1)
        
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
    
    # First pass: extract articles
    articles = []
    current_article_lines = []
    current_article_title = "Phần mở đầu"  # Default title for content before the first article
    start_line = 1  # Start from line 1 for human readability
    current_line = 0
    found_first_article = False
    
    for i, line in enumerate(lines):
        current_line = i + 1  # 1-based line numbering
        
        # Check if this line starts a new article (Điều)
        article_match = re.match(r'^Điều\s+(\d+)[.:]\s+(.*)', line)
        
        if article_match:
            # If we've accumulated content for a previous article, save it
            if current_article_lines:
                articles.append({
                    "title": current_article_title,
                    "content": "\n".join(current_article_lines),
                    "start_line": start_line,
                    "end_line": current_line - 1
                })
            
            # Start a new article
            found_first_article = True
            current_article_title = f"Điều {article_match.group(1)}: {article_match.group(2)}"
            current_article_lines = [line]
            start_line = current_line
        else:
            # Continue with the current article
            current_article_lines.append(line)
    
    # Don't forget the last article
    if current_article_lines:
        articles.append({
            "title": current_article_title,
            "content": "\n".join(current_article_lines),
            "start_line": start_line,
            "end_line": current_line
        })
    
    # Second pass: create overlapping chunks for each article based on sentences
    final_chunks = []
    
    for article_idx, article in enumerate(articles):
        article_content = article["content"]
        article_title = article["title"]
        article_start = article["start_line"]
        article_end = article["end_line"]
        
        # Split article content into sentences
        sentences = split_into_sentences(article_content)
        
        # For very short articles (1-2 sentences), keep them as a single chunk
        if len(sentences) <= max_chunk_sentences:
            # Create a single chunk with all content
            full_content = " ".join(sentences)
            tokenized_content = tokenize_vietnamese_text(full_content)
            
            final_chunks.append({
                "content": tokenized_content,
                "metadata": {
                    "article": article_title,
                    "article_idx": article_idx,
                    "chunk_idx": 0,
                    "total_chunks": 1,
                    "start_line": article_start,
                    "end_line": article_end,
                    "is_complete_article": True,
                    "sentence_range": f"0-{len(sentences)}",
                    "sentences": len(sentences)
                }
            })
        else:
            # For longer articles, create overlapping chunks of sentences
            total_chunks = ((len(sentences) - overlap_sentences) // 
                           (max_chunk_sentences - overlap_sentences) + 1)
            
            # Loop through with the specified stride to create overlapping chunks
            for chunk_idx in range(total_chunks):
                # Calculate start and end indices for this chunk
                start_idx = chunk_idx * (max_chunk_sentences - overlap_sentences)
                end_idx = min(start_idx + max_chunk_sentences, len(sentences))
                
                # If we're at the end with a small remainder, adjust to include
                # the last few sentences properly
                if end_idx - start_idx < max_chunk_sentences and chunk_idx > 0:
                    start_idx = max(0, end_idx - max_chunk_sentences)
                
                # Get chunk sentences and create chunk text
                chunk_sentences = sentences[start_idx:end_idx]
                chunk_text = " ".join(chunk_sentences)
                tokenized_chunk = tokenize_vietnamese_text(chunk_text)
                
                # Generate approximate line numbers (can't know exact due to sentences)
                # This is a rough estimation based on sentence position
                approx_start = article_start
                approx_end = article_end
                
                # Create chunk with metadata
                final_chunks.append({
                    "content": tokenized_chunk,
                    "metadata": {
                        "article": article_title,
                        "article_idx": article_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": total_chunks,
                        "start_line": approx_start,  # Approximate line number 
                        "end_line": approx_end,      # Approximate line number
                        "is_complete_article": False,
                        "sentence_range": f"{start_idx}-{end_idx-1}",
                        "sentences": len(chunk_sentences),
                        "total_sentences": len(sentences)
                    }
                })
    
    print(f"Created {len(final_chunks)} chunks using sentence-based chunking:")
    print(f"  - {max_chunk_sentences} sentences per chunk")
    print(f"  - {overlap_sentences} sentence overlap between chunks")
    
    return final_chunks

def verify_chunking(chunks: List[Dict], total_lines: int) -> bool:
    """
    Verify that all articles from the original text are represented in the chunks.
    
    Args:
        chunks (List[Dict]): The chunks to verify
        total_lines (int): Total number of lines in the original text
        
    Returns:
        bool: True if all articles are accounted for, False otherwise
    """
    # With overlapping chunks, we can't verify line-by-line coverage,
    # but we can check if we have at least one chunk per article
    
    # Check if we have any chunks at all
    if not chunks:
        return False
    
    # Get unique article indices
    article_indices = set(chunk["metadata"]["article_idx"] for chunk in chunks)
    
    # Check if we have at least one chunk for each article
    expected_articles = max(article_indices) + 1 if article_indices else 0
    if len(article_indices) != expected_articles:
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
    
    # Convert the texts to embeddings in smaller batches to avoid memory issues
    batch_size = 8
    embeddings = []
    
    total_batches = (len(texts) + batch_size - 1) // batch_size  # Ceiling division
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{total_batches} ({len(batch_texts)} chunks)")
        
        try:
            # Encode batch
            batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
            print(f"✓ Batch encoded successfully")
        except Exception as e:
            print(f"✗ Error in batch: {e}")
            
            # Process one by one if batch fails
            print("Trying to encode chunks individually...")
            for j, text in enumerate(batch_texts):
                try:
                    # Encode single text
                    emb = model.encode([text], show_progress_bar=False)[0]
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
    # Check that chunks and embeddings match
    if len(chunks) != len(embeddings):
        print(f"Error: Number of chunks ({len(chunks)}) does not match number of embeddings ({len(embeddings)})")
        return
    
    try:
        # Connect to Qdrant
        print(f"Connecting to Qdrant at {host}:{port}...")
        client = QdrantClient(host=host, port=port)
        print("Connected successfully")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        print("Make sure Qdrant is running and accessible")
        return
    
    try:
        # Create collection if it doesn't exist
        vector_size = len(embeddings[0])
        create_collection(client, collection_name, vector_size)
    except Exception as e:
        print(f"Error creating collection: {e}")
        return
    
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
                    "total_sentences": int(metadata.get("total_sentences", 0)),
                    "text_preview": text_preview
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
                client.upsert(
                    collection_name=collection_name,
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

def process_file(
    file_path: str, 
    output_dir: Optional[str] = None,
    store_in_qdrant: bool = True,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    qdrant_collection: str = "hust_documents",
    max_chunk_sentences: int = 2,
    overlap_sentences: int = 1
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
        max_chunk_sentences (int): Maximum number of sentences per chunk (default: 2)
        overlap_sentences (int): Number of overlapping sentences between chunks (default: 1)
        
    Returns:
        List[Dict[str, Any]]: List of chunks with metadata
    """
    print(f"Processing file: {file_path}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Get the total number of lines in the file
    total_lines = len(text.splitlines())
    
    # Chunk the text with specified sizes
    chunks = chunk_by_article(text, max_chunk_sentences=max_chunk_sentences, overlap_sentences=overlap_sentences)
    
    print(f"Created {len(chunks)} chunks using sentence-based chunking:")
    print(f"  - {max_chunk_sentences} sentences per chunk")
    print(f"  - {overlap_sentences} sentence overlap between chunks")
    
    # Get statistics about chunks
    article_stats = {}
    for chunk in chunks:
        article = chunk["metadata"]["article"]
        if article not in article_stats:
            article_stats[article] = 0
        article_stats[article] += 1
    
    print(f"Found {len(article_stats)} unique articles")
    print(f"Average chunks per article: {len(chunks)/len(article_stats):.1f}")
    
    # Verify chunking
    if not verify_chunking(chunks, total_lines):
        print(f"WARNING: Chunking verification failed. Some articles may be missing.")
    else:
        print(f"✓ Verification successful: All articles represented in chunks")
    
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
            import traceback
            traceback.print_exc()
            print("Make sure Qdrant is running and the required dependencies are installed.")
    
    return chunks

def is_qdrant_running(host: str = "localhost", port: int = 6333) -> bool:
    """
    Check if Qdrant server is accessible.
    
    Args:
        host (str): Qdrant server host
        port (int): Qdrant server port
        
    Returns:
        bool: True if Qdrant is accessible, False otherwise
    """
    try:
        client = QdrantClient(host=host, port=port)
        # Try to get collections info
        client.get_collections()
        return True
    except Exception as e:
        print(f"Qdrant server not accessible at {host}:{port}: {e}")
        return False

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
        distance (str): Distance metric to use (Cosine, Euclid, Dot)
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
    
    # Configure chunking parameters
    max_chunk_sentences = 2  # Smaller chunks to avoid encoding issues
    overlap_sentences = 1    # 1 sentence overlap between chunks
    
    # Process the file
    chunks = process_file(
        file_path=file_path, 
        output_dir=chunks_dir,
        store_in_qdrant=store_in_qdrant,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        qdrant_collection=qdrant_collection,
        max_chunk_sentences=max_chunk_sentences,
        overlap_sentences=overlap_sentences
    )
    
    print(f"Successfully processed {len(chunks)} chunks from {os.path.basename(file_path)}")
    
    # Print samples of chunks for verification
    if chunks:
        print("\nSample of first chunk:")
        print(f"Article: {chunks[0]['metadata']['article']}")
        print(f"Sentence range: {chunks[0]['metadata']['sentence_range']}")
        print(f"Content (first 200 chars): {chunks[0]['content'][:200]}...")
        
        # If we have multiple chunks for the same article, show the next one
        if len(chunks) > 1 and chunks[1]['metadata']['article_idx'] == chunks[0]['metadata']['article_idx']:
            print("\nNext chunk from same article:")
            print(f"Article: {chunks[1]['metadata']['article']}")
            print(f"Sentence range: {chunks[1]['metadata']['sentence_range']}")
            print(f"Content (first 200 chars): {chunks[1]['content'][:200]}...")

if __name__ == "__main__":
    main()