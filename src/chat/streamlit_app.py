"""Streamlit chat interface for RAG system."""

import streamlit as st
import os
from typing import List
from ..document_processing.processor import process_uploaded_files
from ..retrieval.chroma_client import add_documents_to_chroma, get_collection_info, clear_collection
from ..rag.workflow import create_rag_workflow, run_rag_query


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_workflow" not in st.session_state:
        st.session_state.rag_workflow = None
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "streamlit_session"


def setup_sidebar():
    """Setup the sidebar with configuration options."""
    st.sidebar.title("RAG Configuration")
    
    # ChromaDB settings
    st.sidebar.subheader("ChromaDB Settings")
    chroma_host = st.sidebar.text_input("ChromaDB Host", value="localhost")
    chroma_port = st.sidebar.number_input("ChromaDB Port", value=8000, min_value=1, max_value=65535)
    collection_name = st.sidebar.text_input("Collection Name", value="rag_documents")
    
    # Model settings
    st.sidebar.subheader("Model Settings")
    model_name = st.sidebar.selectbox(
        "Chat Model",
        ["openai:gpt-4o-mini", "openai:gpt-4o", "openai:gpt-3.5-turbo"],
        index=0
    )
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    
    # Document processing settings
    st.sidebar.subheader("Document Processing")
    chunk_size = st.sidebar.number_input("Chunk Size", value=1000, min_value=100, max_value=5000)
    chunk_overlap = st.sidebar.number_input("Chunk Overlap", value=200, min_value=0, max_value=1000)
    
    return {
        "chroma_host": chroma_host,
        "chroma_port": chroma_port,
        "collection_name": collection_name,
        "model_name": model_name,
        "temperature": temperature,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }


def display_collection_info(config):
    """Display information about the current collection."""
    try:
        info = get_collection_info(
            collection_name=config["collection_name"],
            host=config["chroma_host"],
            port=config["chroma_port"]
        )
        if info:
            st.sidebar.success(f"Collection: {info['name']}")
            st.sidebar.info(f"Documents: {info['count']}")
        else:
            st.sidebar.warning("Collection not found or empty")
    except Exception as e:
        st.sidebar.error(f"Error connecting to ChromaDB: {str(e)}")


def handle_file_upload(config):
    """Handle file upload and processing."""
    st.subheader("Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF or Jupyter Notebook files",
        type=["pdf", "ipynb"],
        accept_multiple_files=True,
        help="Upload PDF files or Jupyter notebooks to add to your knowledge base"
    )
    
    if uploaded_files:
        if st.button("Process and Add Documents"):
            with st.spinner("Processing documents..."):
                try:
                    # Process uploaded files
                    documents = process_uploaded_files(
                        uploaded_files,
                        chunk_size=config["chunk_size"],
                        chunk_overlap=config["chunk_overlap"]
                    )
                    
                    if documents:
                        # Add to ChromaDB
                        success = add_documents_to_chroma(
                            documents,
                            collection_name=config["collection_name"],
                            host=config["chroma_host"],
                            port=config["chroma_port"]
                        )
                        
                        if success:
                            st.success(f"Successfully processed and added {len(documents)} document chunks!")
                            st.session_state.documents_loaded = True
                            # Reset workflow to use new documents
                            st.session_state.rag_workflow = None
                        else:
                            st.error("Failed to add documents to ChromaDB")
                    else:
                        st.warning("No documents were processed")
                        
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")


def initialize_rag_workflow(config):
    """Initialize the RAG workflow if not already done."""
    if st.session_state.rag_workflow is None:
        try:
            with st.spinner("Initializing RAG workflow..."):
                st.session_state.rag_workflow = create_rag_workflow(
                    model_name=config["model_name"],
                    temperature=config["temperature"],
                    collection_name=config["collection_name"],
                    chroma_host=config["chroma_host"],
                    chroma_port=config["chroma_port"]
                )
            st.success("RAG workflow initialized!")
        except Exception as e:
            st.error(f"Error initializing RAG workflow: {str(e)}")
            return False
    return True


def display_chat_interface(config):
    """Display the chat interface."""
    st.subheader("Chat with your Documents")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        if st.session_state.rag_workflow:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = run_rag_query(
                            st.session_state.rag_workflow,
                            prompt,
                            thread_id=st.session_state.thread_id
                        )
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            st.error("RAG workflow not initialized. Please check your configuration.")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="RAG Document Chat",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("RAG Document Chat System")
    st.markdown("Upload your PDFs and Jupyter notebooks, then chat with your documents using AI.")
    
    # Initialize session state
    initialize_session_state()
    
    # Setup sidebar and get configuration
    config = setup_sidebar()
    
    # Display collection info
    display_collection_info(config)
    
    # Clear collection button
    if st.sidebar.button("Clear Collection"):
        if clear_collection(
            collection_name=config["collection_name"],
            host=config["chroma_host"],
            port=config["chroma_port"]
        ):
            st.sidebar.success("Collection cleared!")
            st.session_state.documents_loaded = False
            st.session_state.rag_workflow = None
        else:
            st.sidebar.error("Failed to clear collection")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        handle_file_upload(config)
    
    with col2:
        # Initialize RAG workflow
        if initialize_rag_workflow(config):
            display_chat_interface(config)
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()
