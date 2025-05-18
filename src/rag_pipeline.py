#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import models and retrievers
try:
    from src.models import create_model
    from src.retrievers import create_qdrant_retriever
except ImportError:
    # Handle case when run directly
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import create_model
    from retrievers import create_qdrant_retriever

class RAGPipeline:
    """
    Streamlined RAG Pipeline focusing only on:
    1. Retrieval: Getting relevant documents from Qdrant
    2. Generation: Providing context to LLMs to generate answers
    """
    
    def __init__(self, 
                model_type: str = "openai",
                model_config: Optional[Dict[str, Any]] = None,
                retriever_config: Optional[Dict[str, Any]] = None,
                debug_mode: bool = False):
        """
        Initialize the RAG pipeline.
        
        Args:
            model_type (str): Type of model to use ("openai", "deepseek", "gemini")
            model_config (Dict, optional): Configuration for the model
            retriever_config (Dict, optional): Configuration for the retriever
            debug_mode (bool): Whether to enable debug logging
        """
        self.debug_mode = debug_mode
        
        # Set logging level based on debug mode
        if debug_mode:
            logger.setLevel(logging.DEBUG)
        
        # Initialize components using factory functions
        self.retriever = create_qdrant_retriever(retriever_config)
        logger.info("Initialized Qdrant retriever")
        
        self.model = create_model(model_type, model_config)
        logger.info(f"Initialized {model_type} model")
    
    def answer_question(self, query: str, max_tokens: Optional[int] = None) -> Tuple[str, List[Dict]]:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            query (str): The question to answer
            max_tokens (int, optional): Maximum number of tokens for the answer
            
        Returns:
            Tuple[str, List[Dict]]: The answer and the retrieved contexts
        """
        # Log the query
        logger.info(f"Answering question: {query}")
        
        # Retrieve relevant documents from Qdrant
        start_time = time.time()
        contexts = self.retriever.retrieve(query)
        retrieval_time = time.time() - start_time
        
        logger.info(f"Retrieved {len(contexts)} documents in {retrieval_time:.2f} seconds")
        
        if not contexts:
            logger.warning("No relevant documents found")
            return "Không tìm thấy thông tin liên quan đến câu hỏi.", []
        
        # Format contexts for LLM
        context_text = self._format_contexts_for_llm(contexts)
        
        # Generate answer
        if max_tokens:
            self.model.max_tokens = max_tokens
        
        start_time = time.time()
        answer = self.model.generate_answer(query, context_text)
        generation_time = time.time() - start_time
        
        logger.info(f"Generated answer in {generation_time:.2f} seconds")
        
        return answer, contexts
    
    def _format_contexts_for_llm(self, contexts: List[Dict]) -> str:
        """
        Format contexts for the LLM.
        
        Args:
            contexts (List[Dict]): Retrieved contexts
            
        Returns:
            str: Formatted context text
        """
        formatted_contexts = []
        
        for i, ctx in enumerate(contexts):
            # Extract necessary context info
            content = ctx.get("content", "")
            
            # Handle both payload formats (direct or nested)
            if "payload" in ctx:
                article = ctx["payload"].get("article", "")
                if not content and "content" in ctx["payload"]:
                    content = ctx["payload"]["content"]
                elif "text" in ctx["payload"]:
                    content = ctx["payload"]["text"]
            else:
                article = ctx.get("article", "")
            
            formatted_context = f"[Document {i+1}] {article}\n{content}"
            formatted_contexts.append(formatted_context)
        
        return "\n\n".join(formatted_contexts)


def main():
    """Example usage of the RAG pipeline with hardcoded values."""
    
    # Example model configuration (replace with your API key)
    model_config = {
        "api_key": "your-api-key-here",  # Replace with your actual API key
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    # Example retriever configuration
    retriever_config = {
        "host": "localhost",
        "port": 6333,
        "collection_name": "hust_documents",
        "top_k": 5
    }
    
    pipeline = RAGPipeline(
        model_type="deepseek",
        model_config=model_config,
        retriever_config=retriever_config,
        debug_mode=True
    )
    
    # Example question
    question = "Trường Đại học Bách khoa Hà Nội có những khoa nào?"
    
    # Get answer
    answer, contexts = pipeline.answer_question(question)
    
    # Display answer
    print("\n" + "="*50)
    print("QUESTION:")
    print(question)
    print("\nANSWER:")
    print(answer)
    
    # Display contexts in debug mode
    if pipeline.debug_mode:
        print("\nCONTEXTS:")
        for i, ctx in enumerate(contexts):
            # Handle both payload formats
            if "payload" in ctx:
                article = ctx["payload"].get("article", "")
                content = ctx["payload"].get("content", "")
                if not content:
                    content = ctx["payload"].get("text", "")
            else:
                article = ctx.get("article", "")
                content = ctx.get("content", "")
            
            print(f"\n[{i+1}] {article}")
            print(f"Score: {ctx.get('score', 0):.4f}")
            
            # Truncate content if too long
            if len(content) > 200:
                content = content[:200] + "..."
            print(content)

if __name__ == "__main__":
    main() 