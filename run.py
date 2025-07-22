#!/usr/bin/env python3
"""Run script for the RAG system."""

import subprocess
import sys
import os
from pathlib import Path

def check_chromadb():
    """Check if ChromaDB is running."""
    try:
        import requests
        response = requests.get("http://localhost:8000/api/v1/heartbeat", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def start_streamlit():
    """Start the Streamlit application."""
    print("Starting RAG Document Chat System...")
    
    # Check if ChromaDB is running
    if not check_chromadb():
        print("Warning: ChromaDB is not running on localhost:8000")
        print("Please start ChromaDB with: docker run -p 8000:8000 chromadb/chroma")
        print("Continuing anyway...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("Warning: .env file not found. Please copy .env.example to .env and configure your settings.")
    
    # Run Streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_streamlit()
