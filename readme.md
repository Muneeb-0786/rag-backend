# Any-File RAG System

A versatile Retrieval-Augmented Generation (RAG) system that works with any file format to create embeddings and enable semantic search and question answering.

## Overview

This project provides tools to:
1. Create vector embeddings from various document formats (PDF, TXT, and others)
2. Store these embeddings in vector databases (FAISS or Chroma)
3. Query documents using natural language and get accurate, context-aware responses

## Requirements

### Environment Variables
The following environment variables must be set:
- `COHERE_API_KEY`: Required for embeddings and LLM functionalities
- `GEMINI_API_KEY`: Optional, used for query augmentation and visualization if available

### Dependencies
- Python 3.8+
- langchain and related packages
- Cohere API for embeddings and LLM
- Optional: Google Generative AI (Gemini) for advanced features
- Optional: sentence-transformers for cross-encoder reranking

## Installation

1. Clone this repository
2. Install required dependencies:
   ```
   pip install langchain langchain_community langchain_cohere faiss-cpu chromadb
   pip install sentence-transformers  # Optional, for reranking
   pip install google-generativeai  # Optional, for query augmentation
   ```

## Usage

### Creating Embeddings

Use `create_embeddings.py` to create vector embeddings from your documents:

```bash
# Process a single file
python create_embeddings.py --file path/to/document.pdf

# Process multiple files or directories
python create_embeddings.py --files path/to/doc1.pdf path/to/doc2.txt path/to/folder

# Process files recursively from a directory
python create_embeddings.py --files path/to/folder --recursive

# Create a combined index from multiple files
python create_embeddings.py --files path/to/folder --combine --combined-name my_index

# Customize chunk size and overlap
python create_embeddings.py --files path/to/docs --chunk-size 1500 --chunk-overlap 200

# Use Chroma vector database instead of FAISS
python create_embeddings.py --files path/to/docs --vector-db chroma
```

### Querying Documents

Use `query_embeddings.py` to ask questions about your documents:

```bash
# Basic query against an index
python query_embeddings.py --index indexed_docs/my_index

# Specify retrieval method
python query_embeddings.py --index indexed_docs/my_index --retrieval mmr

# Adjust number of documents to retrieve
python query_embeddings.py --index indexed_docs/my_index --k 6

# Use reranking for better document selection
python query_embeddings.py --index indexed_docs/my_index --use-reranking

# Use query augmentation for improved retrieval
python query_embeddings.py --index indexed_docs/my_index --use-augmentation

# Use both reranking and query augmentation
python query_embeddings.py --index indexed_docs/my_index --use-reranking --use-augmentation
```

### Interactive Query Commands

When in the interactive query interface, you can use the following commands:
- `exit` - Quit the application
- `sources` - Show sources from the last question
- `reset` - Reset chat history

## Advanced Features

### Vector Stores
- FAISS (default): Efficient similarity search
- Chroma: SQL-backed vector database with additional metadata capabilities

### Retrieval Methods
- Similarity search (default): Standard vector similarity retrieval
- MMR (Maximum Marginal Relevance): Balances relevance with diversity
- Similarity with score threshold: Only returns documents above a confidence threshold

### Query Enhancement
- Query augmentation: Generates alternative formulations of your question to improve retrieval
- Cross-encoder reranking: Reranks initial results for better precision
- Contextual compression: Extracts the most relevant parts of retrieved documents

### Visualization
When enabled (with Gemini API), the system generates visualizations of:
- Query-document relationships in vector space
- How augmented queries influence document retrieval
- Which documents were selected as most relevant

## Examples

Create embeddings from a collection of PDF files and combine them into a single index:
```bash
python create_embeddings.py --files documents/*.pdf --combine --combined-name research_papers
```

Query the documents with advanced retrieval:
```bash
python query_embeddings.py --index indexed_docs/research_papers --use-reranking --use-augmentation --k 8
```

## License

[Add your licensing information here]