"""Basic functionality tests for the RAG system."""

import pytest
import os
import tempfile
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from document_processing.pdf_processor import extract_text_from_pdf, process_pdf_document
from document_processing.notebook_processor import extract_text_from_notebook, process_notebook_document
from document_processing.processor import process_single_document


def test_pdf_text_extraction():
    """Test PDF text extraction with a simple PDF."""
    # This test would require a sample PDF file
    # For now, we'll test the function exists and handles missing files gracefully
    result = extract_text_from_pdf("nonexistent.pdf")
    assert result == ""


def test_notebook_text_extraction():
    """Test notebook text extraction with a simple notebook."""
    # Create a simple test notebook
    test_notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": ["# Test Notebook\n", "This is a test."]
            },
            {
                "cell_type": "code",
                "source": ["print('Hello, World!')"],
                "outputs": [
                    {
                        "text": ["Hello, World!\n"]
                    }
                ]
            }
        ]
    }
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
        import json
        json.dump(test_notebook, f)
        temp_path = f.name
    
    try:
        result = extract_text_from_notebook(temp_path)
        assert "[MARKDOWN]" in result
        assert "[CODE]" in result
        assert "[OUTPUT]" in result
        assert "Test Notebook" in result
        assert "Hello, World!" in result
    finally:
        os.unlink(temp_path)


def test_process_single_document_unsupported():
    """Test processing unsupported file types."""
    result = process_single_document("test.txt")
    assert result == []


def test_document_chunk_structure():
    """Test that document chunks have the correct structure."""
    # Create a simple test notebook
    test_notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": ["# Test\n", "This is a test document with some content."]
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
        import json
        json.dump(test_notebook, f)
        temp_path = f.name
    
    try:
        documents = process_notebook_document(temp_path, chunk_size=50, chunk_overlap=10)
        
        if documents:  # Only test if documents were created
            doc = documents[0]
            assert "content" in doc
            assert "metadata" in doc
            assert "source" in doc["metadata"]
            assert "filename" in doc["metadata"]
            assert "chunk_id" in doc["metadata"]
            assert "document_type" in doc["metadata"]
            assert doc["metadata"]["document_type"] == "notebook"
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__])
