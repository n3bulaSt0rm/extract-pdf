from langchain_text_splitters import MarkdownHeaderTextSplitter
import os
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime

class MarkdownChunker:
    """
    A class for chunking markdown text based on headers only.
    Uses LangChain's MarkdownHeaderTextSplitter to split text by headers.
    """
    
    def __init__(
        self,
        headers_to_split_on: Optional[List[Tuple[str, str]]] = None
    ):
        """
        Initialize the MarkdownChunker.
        
        Args:
            headers_to_split_on: List of tuples specifying the headers to split on.
                Each tuple should have (header_pattern, header_name).
                If None, default headers (h1-h3) will be used.
        """
        if headers_to_split_on is None:
            # Default header configuration - split on h1, h2, h3
            self.headers_to_split_on = [
                ("#", "heading1"),
                ("##", "heading2"),
                ("###", "heading3")
            ]
        else:
            self.headers_to_split_on = headers_to_split_on
        
        # Initialize the markdown header splitter
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )
    
    def chunk_text(self, text: str) -> List[Dict]:
        """
        Split the markdown text into chunks based on headers only.
        
        Args:
            text: The markdown text to split.
            
        Returns:
            A list of dictionaries where each dictionary represents a chunk with metadata.
        """
        # Split by headers only
        header_splits = self.header_splitter.split_text(text)
        return header_splits
    
    def save_chunks(self, chunks: List[Dict], output_file: str, source_document: str = None):
        """
        Save chunks to a JSON file.
        
        Args:
            chunks: List of document chunks with metadata.
            output_file: Path to the output file.
            source_document: Path to the source document.
        """
        # Convert chunks to serializable format
        serializable_chunks = []
        for i, chunk in enumerate(chunks):
            # Replace newlines with spaces in content
            content = chunk.page_content.replace("\n", " ")
            
            chunk_data = {
                "chunk_id": i + 1,
                "content": content,
                "metadata": chunk.metadata
            }
            
            # Add source document if provided
            if source_document:
                chunk_data["source"] = source_document
            
            serializable_chunks.append(chunk_data)
        
        # Write directly to file - simpler format without extra metadata
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_chunks, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(chunks)} chunks to {output_file}")


def process_markdown_file(
    input_file: str,
    output_file: str
):
    """
    Process a markdown file and save chunks to a JSON file.
    Only splits by markdown headers.
    
    Args:
        input_file: Path to the input markdown file.
        output_file: Path to save the output chunks.
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create chunker
    chunker = MarkdownChunker()
    
    # Process text
    chunks = chunker.chunk_text(text)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save chunks to file
    chunker.save_chunks(chunks, output_file, source_document=input_file)
    
    return len(chunks)


if __name__ == "__main__":
    # Example usage
    input_file = "src/data/quyche_converted.txt"
    output_file = "src/data/quyche_chunks.json"
    
    num_chunks = process_markdown_file(
        input_file=input_file,
        output_file=output_file
    )
    
    print(f"Successfully processed file and created {num_chunks} chunks.")
    print(f"Output file: {os.path.abspath(output_file)}")
