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