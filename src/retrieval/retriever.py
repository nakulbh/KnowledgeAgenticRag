"""Retriever functions for RAG system."""

from typing import List, Dict, Any, Optional
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import Tool
from .chroma_client import search_documents
from langchain_openai import OpenAIEmbeddings


def create_chroma_retriever_function(
    collection_name: str = "rag_documents",
    host: str = "localhost",
    port: int = 8000,
    n_results: int = 5,
    embedding_model: Optional[Any] = None
):
    """Create a retriever function for ChromaDB.
    
    Args:
        collection_name: Name of the ChromaDB collection
        host: ChromaDB host
        port: ChromaDB port
        n_results: Number of results to return
        embedding_model: Embedding model instance
        
    Returns:
        Retriever function
    """
    def retrieve_documents(query: str) -> str:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            
        Returns:
            Formatted string of relevant documents
        """
        results = search_documents(
            query=query,
            n_results=n_results,
            collection_name=collection_name,
            host=host,
            port=port,
            embedding_model=embedding_model
        )
        
        if not results:
            return "No relevant documents found."
        
        # Format results for RAG
        formatted_docs = []
        for i, result in enumerate(results, 1):
            doc_info = f"Document {i}:\n"
            doc_info += f"Source: {result['metadata'].get('filename', 'Unknown')}\n"
            doc_info += f"Type: {result['metadata'].get('document_type', 'Unknown')}\n"
            doc_info += f"Content: {result['content']}\n"
            doc_info += f"Relevance Score: {result['score']:.3f}\n"
            formatted_docs.append(doc_info)
        
        return "\n" + "="*50 + "\n".join(formatted_docs)
    
    return retrieve_documents


def create_retriever_tool_for_rag(
    collection_name: str = "rag_documents",
    host: str = "localhost",
    port: int = 8000,
    n_results: int = 5,
    embedding_model: Optional[Any] = None
) -> Tool:
    """Create a retriever tool for LangGraph RAG system.
    
    Args:
        collection_name: Name of the ChromaDB collection
        host: ChromaDB host
        port: ChromaDB port
        n_results: Number of results to return
        embedding_model: Embedding model instance
        
    Returns:
        LangChain Tool for document retrieval
    """
    retriever_function = create_chroma_retriever_function(
        collection_name=collection_name,
        host=host,
        port=port,
        n_results=n_results,
        embedding_model=embedding_model
    )
    
    # Create tool using LangChain's Tool class
    retriever_tool = Tool(
        name="retrieve_documents",
        description="Search and return relevant documents from the knowledge base. Use this when you need to find information about specific topics, concepts, or questions that might be answered in the uploaded documents.",
        func=retriever_function
    )
    
    return retriever_tool


def get_relevant_context(
    query: str,
    collection_name: str = "rag_documents",
    host: str = "localhost",
    port: int = 8000,
    n_results: int = 3,
    embedding_model: Optional[Any] = None
) -> str:
    """Get relevant context for a query in a simple format.
    
    Args:
        query: Search query
        collection_name: Name of the ChromaDB collection
        host: ChromaDB host
        port: ChromaDB port
        n_results: Number of results to return
        embedding_model: Embedding model instance
        
    Returns:
        Formatted context string
    """
    results = search_documents(
        query=query,
        n_results=n_results,
        collection_name=collection_name,
        host=host,
        port=port,
        embedding_model=embedding_model
    )
    
    if not results:
        return "No relevant information found in the knowledge base."
    
    # Simple context formatting
    context_parts = []
    for result in results:
        context_parts.append(result['content'])
    
    return "\n\n".join(context_parts)
