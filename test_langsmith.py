#!/usr/bin/env python3
"""Test script to verify LangSmith tracing is working."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_langsmith_setup():
    """Test LangSmith configuration."""
    print("🔍 Testing LangSmith Configuration...")
    
    # Check environment variables
    langsmith_tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    langsmith_endpoint = os.getenv("LANGSMITH_ENDPOINT")
    langsmith_project = os.getenv("LANGSMITH_PROJECT")
    
    print(f"LANGSMITH_TRACING: {langsmith_tracing}")
    print(f"LANGSMITH_API_KEY: {'✅ Set' if langsmith_api_key else '❌ Not set'}")
    print(f"LANGSMITH_ENDPOINT: {langsmith_endpoint}")
    print(f"LANGSMITH_PROJECT: {langsmith_project}")
    
    if langsmith_tracing and langsmith_api_key:
        # Set up LangChain tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
        if langsmith_endpoint:
            os.environ["LANGCHAIN_ENDPOINT"] = langsmith_endpoint
        if langsmith_project:
            os.environ["LANGCHAIN_PROJECT"] = langsmith_project
        
        print("\n✅ LangSmith tracing should be enabled!")
        print(f"🔗 Check your traces at: https://smith.langchain.com/")
        print(f"📊 Project: {langsmith_project}")
        
        # Test a simple LangChain operation
        try:
            from langchain.chat_models import init_chat_model
            
            print("\n🧪 Testing with a simple chat model call...")
            model = init_chat_model("openai:gpt-4o-mini", temperature=0)
            response = model.invoke([{"role": "user", "content": "Hello! This is a test for LangSmith tracing."}])
            print(f"✅ Model response: {response.content[:100]}...")
            print("🔍 This should appear in your LangSmith traces!")
            
        except Exception as e:
            print(f"❌ Error testing model: {e}")
    else:
        print("\n❌ LangSmith tracing is not properly configured")
        if not langsmith_tracing:
            print("   - Set LANGSMITH_TRACING=true")
        if not langsmith_api_key:
            print("   - Set LANGSMITH_API_KEY")

if __name__ == "__main__":
    test_langsmith_setup()
