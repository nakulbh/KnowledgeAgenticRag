"""PDF document processing functions."""

import os
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content as string
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""
    
    return text.strip()


def process_pdf_document(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """Process a PDF document into chunks for RAG.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks with metadata
    """
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
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
    filename = Path(pdf_path).name
    
    for i, chunk in enumerate(chunks):
        doc = {
            "content": chunk,
            "metadata": {
                "source": pdf_path,
                "filename": filename,
                "chunk_id": i,
                "document_type": "pdf",
                "total_chunks": len(chunks)
            }
        }
        documents.append(doc)
    
    return documents


def process_multiple_pdfs(pdf_directory: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """Process multiple PDF files from a directory.
    
    Args:
        pdf_directory: Directory containing PDF files
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of all document chunks from all PDFs
    """
    all_documents = []
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        documents = process_pdf_document(pdf_path, chunk_size, chunk_overlap)
        all_documents.extend(documents)
        print(f"Processed {pdf_file}: {len(documents)} chunks")
    
    return all_documents
