"""
Semantic Text Chunking Module

This module provides functionality to split text into semantically meaningful chunks
using LangChain's text splitters and embedding models for semantic analysis.
"""

import os
import re
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path

# LangChain imports
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SemanticChunker,
    TokenTextSplitter
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.deepseek import ChatDeepseek

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class SemanticTextChunker:
    """
    A class that implements semantic chunking for text documents,
    optimizing for semantic coherence rather than just size.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        add_start_index: bool = True,
        buffer_size: int = 1000,
        separators: List[str] = None
    ):
        """
        Initialize the semantic text chunker.
        
        Args:
            embedding_model_name (str): HuggingFace model name for embeddings
            chunk_size (int): Target size for each text chunk (in characters)
            chunk_overlap (int): Number of characters to overlap between chunks
            add_start_index (bool): Whether to add starting index metadata to each chunk
            buffer_size (int): Buffer size for semantic chunking
            separators (List[str]): Custom separators for splitting text
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.add_start_index = add_start_index
        self.buffer_size = buffer_size
        
        # Initialize embeddings model
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
            
            # Initialize semantic chunker
            self.semantic_chunker = SemanticChunker(
                embeddings=self.embeddings,
                buffer_size=self.buffer_size
            )
            
            # Initialize backup text splitter for fallback
            self.backup_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=separators or ["\n\n", "\n", ". ", " ", ""],
                length_function=len
            )
            
            # Initialize token splitter for longer text
            self.token_splitter = TokenTextSplitter(
                chunk_size=self.chunk_size * 4, 
                chunk_overlap=self.chunk_overlap
            )
            
            self.use_semantic_chunking = True
        except Exception as e:
            print(f"Failed to initialize semantic chunker: {str(e)}")
            print("Falling back to recursive character text splitter.")
            
            # Initialize backup text splitter
            self.backup_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=separators or ["\n\n", "\n", ". ", " ", ""],
                length_function=len
            )
            
            self.use_semantic_chunking = False
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before chunking.
        
        Args:
            text (str): The input text to preprocess
            
        Returns:
            str: The preprocessed text
        """
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove excessive spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Ensure proper sentence spacing (period followed by space)
        text = re.sub(r'\.(?=[A-Z])', '. ', text)
        
        return text
    
    def _create_document(self, text: str, metadata: Dict[str, Any] = None) -> Document:
        """
        Create a LangChain Document from text.
        
        Args:
            text (str): The text content
            metadata (Dict[str, Any], optional): Metadata for the document
            
        Returns:
            Document: A LangChain Document object
        """
        if metadata is None:
            metadata = {}
        
        return Document(page_content=text, metadata=metadata)
    
    def _split_into_documents(
        self, 
        text: str, 
        metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Split a text into LangChain Document objects.
        
        Args:
            text (str): The text to split
            metadata (Dict[str, Any], optional): Metadata for the documents
            
        Returns:
            List[Document]: List of Document objects
        """
        if metadata is None:
            metadata = {}
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # First handle extra long texts with token splitter to avoid memory issues
        if len(processed_text) > self.chunk_size * 10:
            initial_docs = self.token_splitter.create_documents([processed_text], [metadata])
            text_chunks = []
            
            for doc in initial_docs:
                if self.use_semantic_chunking:
                    try:
                        # Try semantic chunking first
                        chunks = self.semantic_chunker.split_text(doc.page_content)
                        for i, chunk in enumerate(chunks):
                            chunk_metadata = metadata.copy()
                            # Add start index to metadata if requested
                            if self.add_start_index:
                                chunk_metadata["start_index"] = processed_text.find(chunk)
                            text_chunks.append(self._create_document(chunk, chunk_metadata))
                    except Exception as e:
                        print(f"Semantic chunking failed: {str(e)}. Using fallback splitter.")
                        # Use backup splitter as fallback
                        fallback_docs = self.backup_splitter.split_documents([doc])
                        text_chunks.extend(fallback_docs)
                else:
                    # Use backup splitter directly
                    fallback_docs = self.backup_splitter.split_documents([doc])
                    text_chunks.extend(fallback_docs)
            
            return text_chunks
        else:
            # For smaller texts, try to use semantic chunking directly
            if self.use_semantic_chunking:
                try:
                    # Try semantic chunking
                    chunks = self.semantic_chunker.split_text(processed_text)
                    text_chunks = []
                    
                    for i, chunk in enumerate(chunks):
                        chunk_metadata = metadata.copy()
                        # Add start index to metadata if requested
                        if self.add_start_index:
                            chunk_metadata["start_index"] = processed_text.find(chunk)
                        text_chunks.append(self._create_document(chunk, chunk_metadata))
                    
                    return text_chunks
                except Exception as e:
                    print(f"Semantic chunking failed: {str(e)}. Using fallback splitter.")
            
            # Use backup splitter as fallback or if semantic chunking is disabled
            doc = self._create_document(processed_text, metadata)
            return self.backup_splitter.split_documents([doc])
    
    def create_chunks(
        self, 
        text: str, 
        metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Create semantically meaningful chunks from a text.
        
        Args:
            text (str): The input text to chunk
            metadata (Dict[str, Any], optional): Metadata to attach to each chunk
            
        Returns:
            List[Document]: List of Document objects representing chunks
        """
        return self._split_into_documents(text, metadata)
    
    def create_chunks_from_file(
        self, 
        file_path: Union[str, Path],
        encoding: str = 'utf-8',
        metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Create semantically meaningful chunks from a text file.
        
        Args:
            file_path (Union[str, Path]): Path to the text file
            encoding (str): Text encoding of the file
            metadata (Dict[str, Any], optional): Metadata to attach to each chunk
            
        Returns:
            List[Document]: List of Document objects representing chunks
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if metadata is None:
            metadata = {}
        
        # Add file metadata
        file_metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "file_type": file_path.suffix.lstrip('.')
        }
        metadata.update(file_metadata)
        
        # Read file content
        with open(file_path, 'r', encoding=encoding) as f:
            text = f.read()
        
        # Create chunks
        return self.create_chunks(text, metadata)
    
    def create_chunks_with_summaries(
        self,
        text: str,
        metadata: Dict[str, Any] = None,
        summarize: bool = True
    ) -> List[Document]:
        """
        Create semantically meaningful chunks with summaries for each chunk.
        
        Args:
            text (str): The input text to chunk
            metadata (Dict[str, Any], optional): Metadata to attach to each chunk
            summarize (bool): Whether to generate summaries for chunks
            
        Returns:
            List[Document]: List of Document objects with summaries in metadata
        """
        chunks = self.create_chunks(text, metadata)
        
        if not summarize or not os.getenv("DEEPSEEK_API_KEY"):
            return chunks
        
        try:
            # Initialize Deepseek model for summarization
            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            deepseek_api_base = os.getenv("DEEPSEEK_API_ENDPOINT")
            
            if not deepseek_api_key or not deepseek_api_base:
                print("Deepseek API credentials not found. Skipping summarization.")
                return chunks
            
            model = ChatDeepseek(
                model_name="deepseek-chat",
                deepseek_api_key=deepseek_api_key,
                deepseek_api_base=deepseek_api_base,
                temperature=0.2,
                max_tokens=256
            )
            
            # Create summarization prompt
            prompt = ChatPromptTemplate.from_template(
                "Summarize the following text in Vietnamese in 1-2 sentences, preserving key information:\n\n{text}"
            )
            
            # Create summarization chain
            chain = prompt | model
            
            # Generate summaries for each chunk
            for chunk in chunks:
                try:
                    response = chain.invoke({"text": chunk.page_content})
                    summary = response.content
                    chunk.metadata["summary"] = summary
                except Exception as e:
                    print(f"Error generating summary: {str(e)}")
                    chunk.metadata["summary"] = ""
            
            return chunks
        
        except Exception as e:
            print(f"Error in summarization process: {str(e)}")
            return chunks


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        chunker = SemanticTextChunker(chunk_size=1000, chunk_overlap=100)
        
        try:
            chunks = chunker.create_chunks_from_file(file_path)
            print(f"Created {len(chunks)} chunks from {file_path}")
            
            for i, chunk in enumerate(chunks):
                print(f"\nChunk {i+1}:")
                print(f"Length: {len(chunk.page_content)} chars")
                print(f"Content (first 100 chars): {chunk.page_content[:100]}...")
                print(f"Metadata: {chunk.metadata}")
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("Usage: python text_chunker.py <file_path>")
