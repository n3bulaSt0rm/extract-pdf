#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys
import nltk
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

try:
    from pyvi import ViTokenizer
except ImportError:
    print("Error: Required package 'pyvi' is not installed. Please run 'pip install pyvi==0.1.1'")
    sys.exit(1)

# Download NLTK punkt for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

class TextChunker:
    """
    A class for chunking Vietnamese text documents using various strategies.
    """
    
    def __init__(self, 
                max_chunk_sentences: int = 2, 
                overlap_sentences: int = 1):
        """
        Initialize the text chunker.
        
        Args:
            max_chunk_sentences (int): Maximum number of sentences per chunk
            overlap_sentences (int): Number of overlapping sentences between chunks
        """
        self.max_chunk_sentences = max_chunk_sentences
        self.overlap_sentences = overlap_sentences
    
    @staticmethod
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
    
    @staticmethod
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
    
    @staticmethod
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
    
    def create_overlapping_chunks(self, sentences: List[str]) -> List[str]:
        """
        Create overlapping chunks from a list of sentences.
        
        Args:
            sentences (List[str]): List of sentences to chunk
            
        Returns:
            List[str]: List of text chunks with overlap
        """
        if not sentences:
            return []
        
        # Ensure valid parameters
        chunk_size = max(1, min(self.max_chunk_sentences, len(sentences)))
        overlap_size = max(0, min(self.overlap_sentences, chunk_size - 1))
        
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
    
    def chunk_by_article(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks where each "Điều" (Article) is split into overlapping chunks based on sentences.
        Each chunk contains a fixed number of sentences with overlap.
        Also captures content before the first article as a separate chunk.
        
        Args:
            text (str): The input text to be chunked
            
        Returns:
            List[Dict[str, Any]]: List of chunks with metadata
        """
        # Remove page indicators, TOC, and other unwanted elements first
        text = self.remove_page_indicators_and_toc(text)
        
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
            sentences = self.split_into_sentences(article_content)
            
            # For very short articles (1-2 sentences), keep them as a single chunk
            if len(sentences) <= self.max_chunk_sentences:
                # Create a single chunk with all content
                full_content = " ".join(sentences)
                tokenized_content = self.tokenize_vietnamese_text(full_content)
                
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
                        "sentence_range": f"0-{len(sentences)-1}",
                        "sentences": len(sentences)
                    }
                })
            else:
                # For longer articles, create overlapping chunks of sentences
                total_chunks = ((len(sentences) - self.overlap_sentences) // 
                               (self.max_chunk_sentences - self.overlap_sentences) + 1)
                
                # Loop through with the specified stride to create overlapping chunks
                for chunk_idx in range(total_chunks):
                    # Calculate start and end indices for this chunk
                    start_idx = chunk_idx * (self.max_chunk_sentences - self.overlap_sentences)
                    end_idx = min(start_idx + self.max_chunk_sentences, len(sentences))
                    
                    # If we're at the end with a small remainder, adjust to include
                    # the last few sentences properly
                    if end_idx - start_idx < self.max_chunk_sentences and chunk_idx > 0:
                        start_idx = max(0, end_idx - self.max_chunk_sentences)
                    
                    # Get chunk sentences and create chunk text
                    chunk_sentences = sentences[start_idx:end_idx]
                    chunk_text = " ".join(chunk_sentences)
                    tokenized_chunk = self.tokenize_vietnamese_text(chunk_text)
                    
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
        print(f"  - {self.max_chunk_sentences} sentences per chunk")
        print(f"  - {self.overlap_sentences} sentence overlap between chunks")
        
        return final_chunks
    
    def chunk_by_fixed_size(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Split text into chunks of fixed size with overlap.
        
        Args:
            text (str): The input text to be chunked
            chunk_size (int): Maximum character length of each chunk
            overlap (int): Number of characters to overlap between chunks
            
        Returns:
            List[Dict[str, Any]]: List of chunks with metadata
        """
        # Clean the text first
        text = self.remove_page_indicators_and_toc(text)
        
        # Ensure valid parameters
        chunk_size = max(200, chunk_size)
        overlap = min(chunk_size // 2, max(50, overlap))
        
        # Tokenize the entire text
        tokenized_text = self.tokenize_vietnamese_text(text)
        
        # Split into chunks with overlap
        chunks = []
        start = 0
        
        # Track position in original text (approximate)
        original_length = len(text)
        tokenized_length = len(tokenized_text)
        
        while start < len(tokenized_text):
            # Get end position for this chunk
            end = start + chunk_size
            if end >= len(tokenized_text):
                end = len(tokenized_text)
            
            # Extract the chunk
            chunk = tokenized_text[start:end]
            
            # Estimate start and end positions in original text
            start_pos_ratio = start / tokenized_length if tokenized_length > 0 else 0
            end_pos_ratio = end / tokenized_length if tokenized_length > 0 else 1
            
            start_pos_original = int(start_pos_ratio * original_length)
            end_pos_original = int(end_pos_ratio * original_length)
            
            # Create chunk with metadata
            chunks.append({
                "content": chunk,
                "metadata": {
                    "chunk_idx": len(chunks),
                    "total_chunks": -1,  # Will be updated after all chunks are created
                    "start_pos": start,
                    "end_pos": end,
                    "start_pos_original": start_pos_original,
                    "end_pos_original": end_pos_original,
                    "chunk_type": "fixed_size"
                }
            })
            
            # Move to next chunk with overlap
            start = end - overlap
            
            # If we're at the end, break to avoid tiny chunks
            if start + chunk_size // 2 >= len(tokenized_text):
                break
        
        # Update total chunks
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)
        
        print(f"Created {len(chunks)} chunks using fixed-size chunking:")
        print(f"  - {chunk_size} characters per chunk")
        print(f"  - {overlap} character overlap between chunks")
        
        return chunks
    
    def chunk_hierarchically(self, text: str) -> List[Dict[str, Any]]:
        """
        Create a hierarchical chunking of the text, with article-level chunks
        and sentence-level subchunks.
        
        Args:
            text (str): The input text to be chunked
            
        Returns:
            List[Dict[str, Any]]: List of chunks with hierarchy information
        """
        # First get article-level chunks
        article_chunks = []
        # Remove page indicators, TOC, and other unwanted elements first
        text = self.remove_page_indicators_and_toc(text)
        
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
        
        # Extract articles
        articles = []
        current_article_lines = []
        current_article_title = "Phần mở đầu"  # Default title for content before the first article
        start_line = 1  # Start from line 1 for human readability
        current_line = 0
        
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
        
        # Process each article into a hierarchy of chunks
        all_chunks = []
        
        for article_idx, article in enumerate(articles):
            article_content = article["content"]
            article_title = article["title"]
            article_start = article["start_line"]
            article_end = article["end_line"]
            
            # Create article-level chunk
            tokenized_article = self.tokenize_vietnamese_text(article_content)
            article_chunk = {
                "content": tokenized_article,
                "metadata": {
                    "article": article_title,
                    "article_idx": article_idx,
                    "chunk_idx": article_idx,
                    "total_chunks": len(articles),
                    "start_line": article_start,
                    "end_line": article_end,
                    "is_complete_article": True,
                    "hierarchy_level": "article",
                    "has_children": True
                }
            }
            all_chunks.append(article_chunk)
            
            # Split article content into sentences
            sentences = self.split_into_sentences(article_content)
            
            # Create sentence-level chunks
            for sent_idx, sentence in enumerate(sentences):
                tokenized_sentence = self.tokenize_vietnamese_text(sentence)
                sent_chunk = {
                    "content": tokenized_sentence,
                    "metadata": {
                        "article": article_title,
                        "article_idx": article_idx,
                        "parent_chunk_idx": article_idx,
                        "chunk_idx": sent_idx,
                        "total_chunks": len(sentences),
                        "start_line": article_start,  # Approximate
                        "end_line": article_start,    # Approximate
                        "hierarchy_level": "sentence",
                        "has_children": False,
                        "parent_id": article_idx
                    }
                }
                all_chunks.append(sent_chunk)
        
        print(f"Created {len(all_chunks)} chunks using hierarchical chunking:")
        print(f"  - {len(articles)} article-level chunks")
        print(f"  - {len(all_chunks) - len(articles)} sentence-level chunks")
        
        return all_chunks
    
    def chunk_text(self, text: str, strategy: str = "article", **kwargs) -> List[Dict[str, Any]]:
        """
        Chunk the text using the specified strategy.
        
        Args:
            text (str): The input text to be chunked
            strategy (str): Chunking strategy ("article", "fixed_size", "hierarchical")
            **kwargs: Additional parameters for specific chunking strategies
            
        Returns:
            List[Dict[str, Any]]: List of chunks with metadata
        """
        if strategy == "article":
            return self.chunk_by_article(text)
        elif strategy == "fixed_size":
            chunk_size = kwargs.get("chunk_size", 1000)
            overlap = kwargs.get("overlap", 200)
            return self.chunk_by_fixed_size(text, chunk_size=chunk_size, overlap=overlap)
        elif strategy == "hierarchical":
            return self.chunk_hierarchically(text)
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")
        
    def process_file(self, file_path: str, strategy: str = "article", **kwargs) -> List[Dict[str, Any]]:
        """
        Process a text file by chunking it according to the specified strategy.
        
        Args:
            file_path (str): Path to the text file to be processed
            strategy (str): Chunking strategy ("article", "fixed_size", "hierarchical")
            **kwargs: Additional parameters for specific chunking strategies
            
        Returns:
            List[Dict[str, Any]]: List of chunks with metadata
        """
        print(f"Processing file: {file_path}")
        
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Chunk the text with the specified strategy
        return self.chunk_text(text, strategy=strategy, **kwargs)


def main():
    """Example usage of TextChunker."""
    # Set the specific file to process
    file_path = "example.txt"
    output_dir = "chunks"
    
    # Create the chunker with custom parameters
    max_chunk_sentences = 3
    overlap_sentences = 1
    
    chunker = TextChunker(
        max_chunk_sentences=max_chunk_sentences,
        overlap_sentences=overlap_sentences
    )
    
    # Process the file
    try:
        chunks = chunker.process_file(
            file_path=file_path,
            strategy="article"
        )
        
        print(f"Successfully processed {len(chunks)} chunks from {file_path}")
        
        # Print a sample chunk
        if chunks:
            print("\nSample of first chunk:")
            print(f"Article: {chunks[0]['metadata']['article']}")
            print(f"Sentence range: {chunks[0]['metadata']['sentence_range']}")
            print(f"Content (first 200 chars): {chunks[0]['content'][:200]}...")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main() 