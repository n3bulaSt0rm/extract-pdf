# Vietnamese RAG System

This is a Vietnamese Retrieval-Augmented Generation (RAG) system optimized for CUDA acceleration with a focus on accuracy. The system can embed and retrieve Vietnamese text chunks with high precision using semantic search and BM25 ranking.

## Features

- CUDA-accelerated embedding and retrieval for high performance
- Hybrid ranking combining semantic similarity and BM25 scoring
- Memory-efficient batching for optimal GPU utilization
- Automatic context expansion for better results
- Half-precision (FP16) optimization when supported

## Setup

### Prerequisites

- Python 3.8+
- PyTorch with CUDA support
- Qdrant vector database (running locally or remote)
- Sentence Transformers

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-directory]
```

2. Install the required dependencies:
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install sentence-transformers qdrant-client numpy
```

3. Set up the directories:
```bash
python setup_dirs.py
```

4. Place your data file in the data directory:
```
src/data/final_chunks_9d631398-eae9-4493-8a48-575cb2b92ab0.json
```

## Usage

### Running the Example

```bash
python run_example.py
```

This will:
1. Initialize the embedding module
2. Load and embed chunks from the data file
3. Store them in Qdrant
4. Run example queries with hybrid ranking
5. Display the results

### Using the API

```python
from src.embedders.text_embedder import VietnameseEmbeddingModule
from src.retrievers.qdrant_retriever import VietnameseQueryModule, RankingConfig

# Initialize embedding module
embedding_module = VietnameseEmbeddingModule(
    qdrant_host="localhost",
    qdrant_port=6333,
    collection_name="vietnamese_chunks_test",
    model_name="bkai-foundation-models/vietnamese-bi-encoder"
)

# Load and embed chunks
embedding_module.load_and_embed_chunks("path/to/chunks.json")

# Initialize query module
config = RankingConfig(
    semantic_weight=0.7,
    bm25_weight=0.3,
    similarity_threshold=0.6,
    min_score_threshold=0.2
)

query_module = VietnameseQueryModule(
    embedding_module=embedding_module,
    config=config
)

# Query
results = query_module.query("Your query here", top_k=5)

# Process results
for result in results:
    print(f"Score: {result['score']}")
    print(f"Content: {result['combined_content']}")
```

## Architecture

The system consists of the following components:

- **CUDA Management**: Handles GPU acceleration and memory management
- **Qdrant Operations**: Manages vector database operations
- **Text Embedding**: Converts text into vector embeddings
- **Query Processing**: Performs semantic search and hybrid ranking

## Performance Optimization

The system includes several optimizations for high performance:

1. **CUDA Acceleration**: Uses GPU for fast vector operations
2. **Memory Management**: Dynamic batch sizing based on text length
3. **FP16 Precision**: Uses half-precision when available
4. **Efficient Vector Operations**: Optimized matrix multiplications
5. **Automatic Cleanup**: Prevents memory leaks and OOM errors 