"""
Test script for the RAG Chatbot API
Run this script to test the functionality after starting the server
"""

import requests
import json
import time
import os
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test API health"""
    print("🔍 Testing API health...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ API is healthy")
            print(f"   - Embedding model loaded: {data.get('embedding_model_loaded', False)}")
            print(f"   - Vector store exists: {data.get('vector_store_exists', False)}")
            return True
        else:
            print("❌ API health check failed")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def create_test_document():
    """Create a test document"""
    test_content = """
    RAG Chatbot Documentation
    
    This is a Retrieval-Augmented Generation (RAG) chatbot system built with FastAPI.
    
    Key Features:
    - Multi-format document support (PDF, DOCX, TXT, CSV, DB, images)
    - OCR capabilities for image text extraction
    - Vector similarity search using FAISS
    - Local LLM for answer generation
    - Memory optimized for RTX 2060 GPU
    
    Pricing Information:
    - Basic plan: $10/month
    - Pro plan: $25/month  
    - Enterprise: Custom pricing
    
    Technical Specifications:
    - Uses TinyLlama-1.1B-Chat model
    - Embedding model: all-MiniLM-L6-v2
    - Chunk size: 512 tokens with 100 token overlap
    - Vector store: FAISS with cosine similarity
    
    Installation:
    1. Install dependencies from requirements.txt
    2. Set up Tesseract OCR
    3. Configure environment variables
    4. Run the FastAPI server
    
    The system can handle various document types and provides contextual answers
    based on the uploaded content.
    """
    
    test_file_path = Path("sample_files/test_document.txt")
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(test_content)
    
    print(f"📄 Created test document: {test_file_path}")
    return test_file_path

def test_upload(file_path):
    """Test document upload"""
    print(f"📤 Testing document upload: {file_path.name}")
    
    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "text/plain")}
            response = requests.post(f"{API_BASE_URL}/upload", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Upload successful")
            print(f"   - Chunks created: {data.get('chunks_created', 0)}")
            print(f"   - Processing time: {data.get('processing_time', 0):.2f}s")
            return True
        else:
            print(f"❌ Upload failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False

def test_query(question):
    """Test document query"""
    print(f"🤔 Testing query: '{question}'")
    
    try:
        data = {"question": question}
        response = requests.post(f"{API_BASE_URL}/query", json=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Query successful")
            print(f"   - Answer: {result.get('answer', 'No answer')[:100]}...")
            print(f"   - Confidence: {result.get('confidence_score', 0):.2%}")
            print(f"   - Sources: {', '.join(result.get('source_files', []))}")
            print(f"   - Processing time: {result.get('processing_time', 0):.2f}s")
            return True
        else:
            print(f"❌ Query failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Query error: {e}")
        return False

def test_stats():
    """Test stats endpoint"""
    print("📊 Testing stats...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Stats retrieved")
            print(f"   - Total documents: {data.get('total_documents', 0)}")
            print(f"   - Total chunks: {data.get('total_chunks', 0)}")
            print(f"   - Vector store size: {data.get('vector_store_size', 0)} bytes")
            return True
        else:
            print(f"❌ Stats failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Stats error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Starting RAG Chatbot API Tests")
    print("=" * 50)
    
    # Test 1: Health check
    if not test_health():
        print("\n❌ Health check failed. Please start the server first.")
        print("Run: python main.py")
        return
    
    print()
    
    # Test 2: Create and upload test document
    test_file = create_test_document()
    
    print()
    
    if not test_upload(test_file):
        print("\n❌ Upload test failed.")
        return
    
    print()
    
    # Give the system a moment to process
    time.sleep(2)
    
    # Test 3: Query tests
    test_questions = [
        "What is this chatbot system?",
        "What are the pricing plans?", 
        "What are the key features?",
        "How do I install this system?",
        "What models does it use?"
    ]
    
    for question in test_questions:
        test_query(question)
        print()
        time.sleep(1)  # Small delay between queries
    
    # Test 4: Stats
    test_stats()
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed!")
    print("\n💡 Next steps:")
    print("1. Try the Streamlit frontend: streamlit run streamlit_app.py")
    print("2. Upload your own documents via the API or web interface")
    print("3. Experiment with different question types")

if __name__ == "__main__":
    main()
