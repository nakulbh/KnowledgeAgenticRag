# RAG Document Chat System

A Retrieval-Augmented Generation (RAG) system that allows you to upload PDF files and Jupyter notebooks, then chat with your documents using AI. Built with LangGraph, ChromaDB, and Streamlit.

## Features

- **Document Processing**: Upload and process PDF files and Jupyter notebooks
- **Vector Storage**: Uses ChromaDB for efficient document storage and retrieval
- **Agentic RAG**: Implements LangGraph-based conversational RAG with memory
- **Chat Interface**: Simple Streamlit interface for document upload and chat
- **Conversation History**: Maintains chat history using MemorySaver checkpointer
- **Document Grading**: Automatically grades document relevance and rewrites queries if needed

## Architecture

The system follows a functional approach without classes and includes:

- **Document Processing**: Extract text from PDFs and notebooks, split into chunks
- **Vector Storage**: ChromaDB integration for embeddings and similarity search
- **Agentic Workflow**: LangGraph workflow with query generation, document grading, and answer generation
- **Chat Interface**: Streamlit app with file upload and conversational chat

## Prerequisites

- Python 3.13+
- ChromaDB running in a container on port 8000
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd resourceAgregationRag
```

2. Install dependencies:
```bash
pip install -e .
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key and other settings
```

4. Start ChromaDB container:
```bash
docker run -p 8000:8000 chromadb/chroma
```

## Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

### Using the Interface

1. **Configure Settings**: Use the sidebar to configure ChromaDB connection, model settings, and document processing parameters

2. **Upload Documents**: Upload PDF files or Jupyter notebooks using the file uploader

3. **Process Documents**: Click "Process and Add Documents" to extract text, create chunks, and store in ChromaDB

4. **Chat**: Ask questions about your documents in the chat interface

### Programmatic Usage

```python
from src.document_processing.processor import process_single_document
from src.retrieval.chroma_client import add_documents_to_chroma
from src.rag.workflow import create_rag_workflow, run_rag_query

# Process a document
documents = process_single_document("path/to/document.pdf")

# Add to ChromaDB
add_documents_to_chroma(documents)

# Create RAG workflow
workflow = create_rag_workflow()

# Query the system
response = run_rag_query(workflow, "What is this document about?")
print(response)
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `CHROMA_HOST`: ChromaDB host (default: localhost)
- `CHROMA_PORT`: ChromaDB port (default: 8000)
- `DEFAULT_MODEL`: Default chat model (default: openai:gpt-4o-mini)
- `DEFAULT_TEMPERATURE`: Model temperature (default: 0.0)
- `DEFAULT_CHUNK_SIZE`: Document chunk size (default: 1000)
- `DEFAULT_CHUNK_OVERLAP`: Chunk overlap (default: 200)

### Supported File Types

- PDF files (.pdf)
- Jupyter notebooks (.ipynb)

## Project Structure

```
resourceAgregationRag/
├── src/
│   ├── document_processing/    # Document processing functions
│   ├── retrieval/             # ChromaDB integration
│   ├── rag/                   # LangGraph workflow
│   ├── chat/                  # Streamlit interface
│   └── utils/                 # Utility functions
├── config/                    # Configuration files
├── data/                      # Data directories
├── docs/                      # Documentation
├── tests/                     # Test files
├── app.py                     # Main application entry point
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## Troubleshooting

### ChromaDB Connection Issues

- Ensure ChromaDB container is running on the correct port
- Check firewall settings
- Verify host and port configuration

### Document Processing Issues

- Ensure uploaded files are valid PDFs or Jupyter notebooks
- Check file permissions
- Verify sufficient disk space for temporary files

### Model Issues

- Verify OpenAI API key is set correctly
- Check API quota and billing
- Ensure model name is correct