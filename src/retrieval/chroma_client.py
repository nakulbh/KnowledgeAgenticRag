"""ChromaDB client functions for vector storage and retrieval."""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import uuid
from langchain_openai import OpenAIEmbeddings


def get_chroma_client(host: str = "localhost", port: int = 8000) -> chromadb.HttpClient:
    """Get ChromaDB client connection.
    
    Args:
        host: ChromaDB host
        port: ChromaDB port
        
    Returns:
        ChromaDB client instance
    """
    try:
        client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(allow_reset=True)
        )
        # Test connection
        client.heartbeat()
        return client
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        raise


def create_or_get_collection(client: chromadb.HttpClient, collection_name: str = "rag_documents"):
    """Create or get a ChromaDB collection.
    
    Args:
        client: ChromaDB client
        collection_name: Name of the collection
        
    Returns:
        ChromaDB collection instance
    """
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG document collection"}
        )
        return collection
    except Exception as e:
        print(f"Error creating/getting collection: {e}")
        raise


def add_documents_to_chroma(
    documents: List[Dict[str, Any]], 
    collection_name: str = "rag_documents",
    host: str = "localhost", 
    port: int = 8000,
    embedding_model: Optional[Any] = None
) -> bool:
    """Add documents to ChromaDB collection.
    
    Args:
        documents: List of document chunks with content and metadata
        collection_name: Name of the collection
        host: ChromaDB host
        port: ChromaDB port
        embedding_model: Embedding model instance
        
    Returns:
        Success status
    """
    try:
        client = get_chroma_client(host, port)
        collection = create_or_get_collection(client, collection_name)
        
        # Initialize embedding model if not provided
        if embedding_model is None:
            embedding_model = OpenAIEmbeddings()
        
        # Prepare data for ChromaDB
        texts = []
        metadatas = []
        ids = []
        
        for doc in documents:
            texts.append(doc["content"])
            metadatas.append(doc["metadata"])
            ids.append(str(uuid.uuid4()))
        
        # Generate embeddings
        embeddings = embedding_model.embed_documents(texts)
        
        # Add to collection
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(documents)} documents to ChromaDB collection '{collection_name}'")
        return True
        
    except Exception as e:
        print(f"Error adding documents to ChromaDB: {e}")
        return False


def search_documents(
    query: str, 
    n_results: int = 5,
    collection_name: str = "rag_documents",
    host: str = "localhost", 
    port: int = 8000,
    embedding_model: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """Search for relevant documents in ChromaDB.
    
    Args:
        query: Search query
        n_results: Number of results to return
        collection_name: Name of the collection
        host: ChromaDB host
        port: ChromaDB port
        embedding_model: Embedding model instance
        
    Returns:
        List of relevant documents with metadata and scores
    """
    try:
        client = get_chroma_client(host, port)
        collection = create_or_get_collection(client, collection_name)
        
        # Initialize embedding model if not provided
        if embedding_model is None:
            embedding_model = OpenAIEmbeddings()
        
        # Generate query embedding
        query_embedding = embedding_model.embed_query(query)
        
        # Search in collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            result = {
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]  # Convert distance to similarity score
            }
            formatted_results.append(result)
        
        return formatted_results
        
    except Exception as e:
        print(f"Error searching documents in ChromaDB: {e}")
        return []


def get_collection_info(
    collection_name: str = "rag_documents",
    host: str = "localhost", 
    port: int = 8000
) -> Dict[str, Any]:
    """Get information about a ChromaDB collection.
    
    Args:
        collection_name: Name of the collection
        host: ChromaDB host
        port: ChromaDB port
        
    Returns:
        Collection information
    """
    try:
        client = get_chroma_client(host, port)
        collection = create_or_get_collection(client, collection_name)
        
        count = collection.count()
        
        return {
            "name": collection_name,
            "count": count,
            "metadata": collection.metadata
        }
        
    except Exception as e:
        print(f"Error getting collection info: {e}")
        return {}


def clear_collection(
    collection_name: str = "rag_documents",
    host: str = "localhost", 
    port: int = 8000
) -> bool:
    """Clear all documents from a ChromaDB collection.
    
    Args:
        collection_name: Name of the collection
        host: ChromaDB host
        port: ChromaDB port
        
    Returns:
        Success status
    """
    try:
        client = get_chroma_client(host, port)
        client.delete_collection(collection_name)
        print(f"Cleared collection '{collection_name}'")
        return True
        
    except Exception as e:
        print(f"Error clearing collection: {e}")
        return False
