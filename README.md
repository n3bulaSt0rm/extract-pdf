# Vietnamese Text Processing and RAG Pipeline

A modular Retrieval-Augmented Generation (RAG) pipeline for Vietnamese text processing. The system separates concerns with dedicated modules for text extraction, chunking, embedding, retrieval, and answer generation.

## Project Structure

```
.
├── src/                # Source code
│   ├── chunkers/       # Text chunking components
│   │   ├── text_chunker.py          # Main chunking implementation
│   │   └── overlap-sentences/       # Specialized sentence-based chunking
│   ├── embedders/      # Text embedding components  
│   │   └── text_embedder.py         # Embedding and storage implementation
│   ├── extractors/     # Document extraction components
│   │   └── docling/                 # Vietnamese PDF extraction using Docling
│   ├── models/         # LLM integration components
│   ├── retrievers/     # Document retrieval components
│   │   ├── vector_retriever.py      # Vector-based document retrieval
│   │   ├── retriever_evaluation.py  # Evaluation metrics for retrieval
│   │   └── query_qdrant.py          # Direct Qdrant querying
│   ├── rag.py          # Original RAG pipeline implementation
│   └── rag_pipeline.py # Modular RAG pipeline focusing on retrieval and generation
├── data/               # Sample data and documents
├── output/             # Output directory
│   ├── chunks/         # Chunked documents
│   ├── embeddings/     # Stored embeddings
│   └── evaluation/     # Evaluation results
└── README.md           # This file
```

## Components

The system has a clear separation of concerns with dedicated modules:

### Data Processing (Pre-RAG)
- **Extractors**: Extract text from documents (PDF, etc.)
- **Chunkers**: Split text into chunks for embedding and retrieval

### RAG Pipeline Core Components
- **Embedders**: Convert text to vector representations
- **Storage**: Store vectors in Qdrant database
- **Retrievers**: Find relevant documents based on queries
- **Models**: Generate answers using LLMs (OpenAI, DeepSeek, Gemini)

## RAG Pipeline

The RAG pipeline focuses specifically on three core components:

1. **Storage**: Vector database for document embeddings
2. **Retrieval**: Finding the most relevant documents for a query
3. **Generation**: Using LLMs to produce answers based on retrieved contexts

The pipeline supports multiple LLM providers:
- **OpenAI**: GPT-3.5/GPT-4 models
- **DeepSeek**: DeepSeek models
- **Gemini**: Google's Gemini models

## Usage

### Installation

1. Set up environment:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. Install Qdrant for vector storage:

```bash
# Using Docker
docker run -p 6333:6333 -p 6334:6334 -v ./qdrant_data:/qdrant/storage qdrant/qdrant
```

### Processing Workflow

1. **Text Extraction**: Extract text from documents
2. **Chunking**: Divide text into appropriate chunks 
3. **Embedding**: Convert chunks to vector representations
4. **Storage**: Store vectors in Qdrant
5. **Retrieval & Generation**: Answer questions using RAG pipeline

### RAG Pipeline Usage

The new RAG pipeline focuses on answering questions using retrieved contexts:

```bash
# Answer a question using the RAG pipeline
python src/rag_pipeline.py --query "What are the eligibility criteria?"

# Specify which LLM to use
python src/rag_pipeline.py --query "What are the eligibility criteria?" --model openai

# Enable debug mode to see retrieved contexts
python src/rag_pipeline.py --query "What are the eligibility criteria?" --debug

# Use custom configuration
python src/rag_pipeline.py --query "What are the eligibility criteria?" --config path/to/config.json
```

### Configuration

Create a JSON configuration file to customize the pipeline:

```json
{
  "embedder": {
    "model_name": "vinai/phobert-base",
    "batch_size": 8
  },
  "storage": {
    "host": "localhost",
    "port": 6333,
    "collection_name": "hust_documents",
    "distance": "Cosine"
  },
  "retriever": {
    "similarity_metric": "cosine",
    "top_k": 5,
    "use_reranking": true
  },
  "model": {
    "type": "openai",
    "api_key": "your-api-key",
    "model_name": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 1024
  },
  "output": {
    "base_dir": "output",
    "embeddings_dir": "embeddings",
    "evaluation_dir": "evaluation"
  }
}
```

### Programmatic Usage

```python
from src.rag_pipeline import RAGPipeline

# Create pipeline
pipeline = RAGPipeline(config_path="config.json")

# Answer a question
answer, contexts = pipeline.answer_question("What is the eligibility criteria?")
print(answer)

# Evaluate retriever performance
evaluation_data = {
    "queries": ["Query 1", "Query 2"],
    "relevant_docs": [{"doc1", "doc3"}, {"doc2", "doc4"}]
}
metrics = pipeline.evaluate_retriever(evaluation_data)
```

## Evaluation

The pipeline includes comprehensive evaluation capabilities for retrieval:

- **Precision, Recall, F1**: Measures retrieval accuracy
- **MRR (Mean Reciprocal Rank)**: Evaluates ranking quality
- **NDCG (Normalized Discounted Cumulative Gain)**: Measures ranking quality with relevance weights
- **MAP (Mean Average Precision)**: Evaluates precision across recall levels
- **Coverage & Diversity**: Assesses result completeness and variety
- **Query Efficiency**: Measures retrieval speed

## Requirements

- Python 3.8+
- PyTorch
- Sentence Transformers
- Qdrant
- PyVi (Vietnamese NLP tools)
- NLTK
- Pandas & Matplotlib (for evaluation)
- LLM API access (OpenAI, DeepSeek, or Gemini) 