"""Main document processing functions."""

import os
from typing import List, Dict, Any, Union
from .pdf_processor import process_pdf_document, process_multiple_pdfs
from .notebook_processor import process_notebook_document, process_multiple_notebooks


def process_single_document(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """Process a single document (PDF or notebook).
    
    Args:
        file_path: Path to the document file
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks with metadata
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return process_pdf_document(file_path, chunk_size, chunk_overlap)
    elif file_extension == '.ipynb':
        return process_notebook_document(file_path, chunk_size, chunk_overlap)
    else:
        print(f"Unsupported file type: {file_extension}")
        return []


def process_documents_from_directory(directory_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """Process all supported documents from a directory.
    
    Args:
        directory_path: Directory containing documents
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of all document chunks from all supported files
    """
    all_documents = []
    
    # Process PDFs
    pdf_documents = process_multiple_pdfs(directory_path, chunk_size, chunk_overlap)
    all_documents.extend(pdf_documents)
    
    # Process notebooks
    notebook_documents = process_multiple_notebooks(directory_path, chunk_size, chunk_overlap)
    all_documents.extend(notebook_documents)
    
    return all_documents


def process_uploaded_files(uploaded_files: List[Union[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """Process uploaded files from Streamlit.
    
    Args:
        uploaded_files: List of uploaded file objects or paths
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks with metadata
    """
    all_documents = []
    
    for uploaded_file in uploaded_files:
        if hasattr(uploaded_file, 'name'):
            # Streamlit uploaded file object
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            if file_extension == '.pdf':
                # Save temporarily and process
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                documents = process_pdf_document(temp_path, chunk_size, chunk_overlap)
                all_documents.extend(documents)
                
                # Clean up temp file
                os.remove(temp_path)
                
            elif file_extension == '.ipynb':
                # Save temporarily and process
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                documents = process_notebook_document(temp_path, chunk_size, chunk_overlap)
                all_documents.extend(documents)
                
                # Clean up temp file
                os.remove(temp_path)
        else:
            # File path string
            documents = process_single_document(uploaded_file, chunk_size, chunk_overlap)
            all_documents.extend(documents)
    
    return all_documents
