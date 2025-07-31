"""
Streamlit Frontend for RAG Chatbot
Provides a user-friendly interface for document upload and querying
"""

import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image
import time
import json

# Configure page
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_document(file):
    """Upload document to the API"""
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def query_documents(question, image_base64=None):
    """Query the RAG system"""
    try:
        data = {"question": question}
        if image_base64:
            data["image_base64"] = image_base64
        
        response = requests.post(f"{API_BASE_URL}/query", json=data)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_stats():
    """Get system statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def clear_vector_store():
    """Clear the vector store"""
    try:
        response = requests.delete(f"{API_BASE_URL}/clear")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def image_to_base64(image):
    """Convert PIL image to base64"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def main():
    st.title("🤖 RAG Chatbot")
    st.markdown("Upload documents and ask questions about their content!")
    
    # Check API health in a container
    with st.container():
        col_status, col_empty = st.columns([1, 2])
        with col_status:
            if not check_api_health():
                st.error("❌ API not running")
                st.code("python main.py", language="bash")
                return
            else:
                st.success("✅ API is running")
    
    # Sidebar for file upload and stats
    with st.sidebar:
        st.header("📄 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'txt', 'csv', 'db', 'jpg', 'jpeg', 'png'],
            help="Supported formats: PDF, DOCX, TXT, CSV, DB, JPG, PNG"
        )
        
        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                result = upload_document(uploaded_file)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.success(f"✅ {result['message']}")
                    st.info(f"Created {result['chunks_created']} chunks")
                    st.info(f"Processing time: {result['processing_time']:.2f}s")
        
        st.divider()
        
        # System statistics
        st.header("📊 System Stats")
        if st.button("Refresh Stats"):
            stats = get_stats()
            if "error" not in stats:
                st.metric("Documents", stats.get('total_documents', 0))
                st.metric("Text Chunks", stats.get('total_chunks', 0))
                st.metric("Vector Store Size", f"{stats.get('vector_store_size', 0)} bytes")
        
        st.divider()
        
        # Clear vector store
        if st.button("🗑️ Clear All Documents", type="secondary"):
            if st.checkbox("I understand this will delete all uploaded documents"):
                result = clear_vector_store()
                if "error" not in result:
                    st.success("✅ Vector store cleared")
                else:
                    st.error(f"Error: {result['error']}")
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.header("ℹ️ How to Use")
        st.markdown("""
        1. **Upload Documents**: Use the sidebar to upload your documents
        2. **Ask Questions**: Type your question in the text area
        3. **Optional OCR**: Upload an image to extract text from it
        4. **Get Answers**: Click "Ask Question" to get AI-powered responses
        
        **Supported Formats:**
        - 📄 PDF files
        - 📝 Word documents (DOCX)
        - 📃 Text files (TXT)
        - 📊 CSV files
        - 🗄️ SQLite databases (DB)
        - 🖼️ Images (JPG, PNG) with OCR
        
        **Tips:**
        - Upload multiple documents for better context
        - Ask specific questions for better results
        - Use images to extract text via OCR
        """)
        
        st.divider()
        
        st.header("🔧 System Info")
        st.code(f"API URL: {API_BASE_URL}")
        
        # Display current stats
        stats = get_stats()
        if "error" not in stats:
            st.json(stats)
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Question input
        question = st.text_area(
            "Enter your question:",
            placeholder="What does the document say about pricing?",
            height=100
        )
        
        # Optional image upload for OCR
        with st.expander("📸 Upload Image for OCR (Optional)"):
            uploaded_image = st.file_uploader(
                "Upload an image to extract text",
                type=['jpg', 'jpeg', 'png'],
                key="image_upload"
            )
            
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded image", use_container_width=True)
        
        # Query button
        if st.button("🔍 Ask Question", type="primary", disabled=not question.strip()):
            with st.spinner("Thinking..."):
                image_base64 = None
                if uploaded_image:
                    image = Image.open(uploaded_image)
                    image_base64 = image_to_base64(image)
                
                start_time = time.time()
                result = query_documents(question, image_base64)
                processing_time = time.time() - start_time
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    # Display answer
                    st.subheader("🤖 Answer")
                    st.write(result['answer'])
                    
                    # Display metadata
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Confidence", f"{result['confidence_score']:.2%}")
                    with col_b:
                        st.metric("Sources", len(result['source_files']))
                    with col_c:
                        st.metric("Response Time", f"{processing_time:.2f}s")
                    
                    # Display sources
                    if result['source_files']:
                        st.subheader("📚 Sources")
                        for source in result['source_files']:
                            st.write(f"• {source}")

if __name__ == "__main__":
    main()
