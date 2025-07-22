"""Jupyter notebook processing functions."""

import json
import os
from typing import List, Dict, Any
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_notebook(notebook_path: str) -> str:
    """Extract text content from a Jupyter notebook.
    
    Args:
        notebook_path: Path to the notebook file
        
    Returns:
        Extracted text content as string
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as file:
            notebook = json.load(file)
        
        text_content = []
        
        for cell in notebook.get('cells', []):
            cell_type = cell.get('cell_type', '')
            source = cell.get('source', [])
            
            if cell_type == 'markdown':
                # Add markdown content
                if isinstance(source, list):
                    content = ''.join(source)
                else:
                    content = source
                text_content.append(f"[MARKDOWN]\n{content}\n")
                
            elif cell_type == 'code':
                # Add code content
                if isinstance(source, list):
                    content = ''.join(source)
                else:
                    content = source
                text_content.append(f"[CODE]\n{content}\n")
                
                # Add output if available
                outputs = cell.get('outputs', [])
                for output in outputs:
                    if 'text' in output:
                        output_text = output['text']
                        if isinstance(output_text, list):
                            output_text = ''.join(output_text)
                        text_content.append(f"[OUTPUT]\n{output_text}\n")
        
        return '\n'.join(text_content)
        
    except Exception as e:
        print(f"Error extracting text from notebook {notebook_path}: {e}")
        return ""


def process_notebook_document(notebook_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """Process a Jupyter notebook into chunks for RAG.
    
    Args:
        notebook_path: Path to the notebook file
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks with metadata
    """
    # Extract text from notebook
    text = extract_text_from_notebook(notebook_path)
    
    if not text:
        return []
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    
    # Create document chunks with metadata
    documents = []
    filename = Path(notebook_path).name
    
    for i, chunk in enumerate(chunks):
        doc = {
            "content": chunk,
            "metadata": {
                "source": notebook_path,
                "filename": filename,
                "chunk_id": i,
                "document_type": "notebook",
                "total_chunks": len(chunks)
            }
        }
        documents.append(doc)
    
    return documents


def process_multiple_notebooks(notebook_directory: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """Process multiple notebook files from a directory.
    
    Args:
        notebook_directory: Directory containing notebook files
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of all document chunks from all notebooks
    """
    all_documents = []
    notebook_files = [f for f in os.listdir(notebook_directory) if f.lower().endswith('.ipynb')]
    
    for notebook_file in notebook_files:
        notebook_path = os.path.join(notebook_directory, notebook_file)
        documents = process_notebook_document(notebook_path, chunk_size, chunk_overlap)
        all_documents.extend(documents)
        print(f"Processed {notebook_file}: {len(documents)} chunks")
    
    return all_documents
