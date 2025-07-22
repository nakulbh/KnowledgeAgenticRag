"""
Configuration constants for the RAG system.
Contains all configurable parameters for document processing, embeddings, and retrieval.
"""

# Document Processing Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS_PER_CHUNK = 500

# PDF Processing
PDF_EXTRACT_IMAGES = False
PDF_EXTRACT_TABLES = True

# Jupyter Notebook Processing
INCLUDE_CODE_CELLS = True
INCLUDE_MARKDOWN_CELLS = True
INCLUDE_OUTPUT_CELLS = False

# Embedding Model Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
BATCH_SIZE_EMBEDDINGS = 100

# Vector Store Configuration
VECTOR_STORE_TYPE = "faiss"  # Options: "faiss", "chroma", "inmemory"
SIMILARITY_THRESHOLD = 0.7
MAX_RETRIEVAL_RESULTS = 5

# Language Model Configuration
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.1
MAX_TOKENS_RESPONSE = 1000

# File Processing
SUPPORTED_PDF_EXTENSIONS = [".pdf"]
SUPPORTED_NOTEBOOK_EXTENSIONS = [".ipynb"]
MAX_FILE_SIZE_MB = 50

# Paths
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
VECTOR_STORE_PATH = "data/processed/vector_store"
METADATA_PATH = "data/processed/metadata.json"

# Retrieval Configuration
RERANK_RESULTS = True
CONTEXT_WINDOW_SIZE = 4000
QUERY_EXPANSION = True

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "logs/rag_system.log"
