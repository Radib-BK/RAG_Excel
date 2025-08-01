# 🤖 RAG Chatbot - Complete Document Q&A System

A powerful Retrieval-Augmented Generation (RAG) chatbot that can process multiple document types, extract text using OCR, and answer questions using locally hosted AI models. Optimized for RTX 2060 GPUs but works on any hardware.

## ✨ Key Features

- 📄 **Multi-format Support**: PDF, DOCX, TXT, CSV, SQLite DB, JPG, PNG
- 🔍 **OCR Text Extraction**: Extract text from images using Tesseract
- 🧠 **Semantic Chunking**: LangChain-style recursive text splitting preserves document structure
- 📊 **Enhanced Table Processing**: Auto-detection and structured extraction of tables from PDFs
- 🎯 **Accurate Tokenization**: Real tokenizer integration for precise token counting
- 🧠 **Local AI Models**: TinyLlama LLM + SentenceTransformers embeddings
- ⚡ **GPU Optimized**: RTX 2060 ready with 4-bit quantization
- 🌐 **Web Interface**: FastAPI backend + Streamlit frontend
- 🐳 **Docker Ready**: Simple containerized deployment
- 💾 **Persistent Storage**: FAISS vector database with rich metadata

## 🚀 Quick Start Guide

### **⚠️ IMPORTANT: Before You Start**

**You MUST create a `.env` file** before running the application. This file contains essential configuration including your HuggingFace token.

**Quick Setup:**
1. 🔑 Get a **free** HuggingFace token: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. 📁 Create `.env` file in the project root
3. 📝 Copy the configuration from Step 2 below

---

### **Method 1: Local Development (Recommended for GPU Users)**

#### **Step 1: Clone and Setup**
```cmd
# Clone the repository
git clone https://github.com/Radib-BK/RAG_Excel.git
cd RAG_Excel

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate
```

#### **Step 2: Create Environment File (REQUIRED)**

**Create a `.env` file in the project root directory:**

```cmd
# Create .env file (Windows)
echo. > .env

# OR create manually with any text editor
notepad .env
```

**Add this content to your `.env` file:**
```env
# Environment variables for the RAG system
HUGGINGFACE_TOKEN=your_token
MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_STORE_PATH=./vector_store
CHUNK_SIZE=512
CHUNK_OVERLAP=100
TOP_K_RESULTS=5
MAX_TOKENS=150
DEVICE=cuda
```

**🔑 Important Notes:**
- Replace `your token` with your own [Hugging Face token](https://huggingface.co/settings/tokens)
- The token is **free** but required for downloading models
- Keep your token private and don't share it publicly

#### **Step 3: Install Dependencies**

**For GPU Users (RTX 2060/RTX series):**
```cmd
# Install PyTorch with CUDA support first (REQUIRED for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

**For CPU-Only Users:**
```cmd
# Install all requirements (PyTorch CPU version included)
pip install -r requirements.txt
```

**Setup Models & Directories:**
```cmd
# Run installation script (downloads models, creates directories)
python install.py
```

#### **Step 4: Install Tesseract OCR**
- **Windows**: Download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **Add to PATH**: `C:\Program Files\Tesseract-OCR`

#### **Step 5: Start the Application**

**Option A: Automatic Launcher (Recommended):**
```cmd
cd RAG_Excel
.venv\Scripts\activate
python run.py
```
*This automatically starts both FastAPI backend and Streamlit frontend*

**Option B: Manual (2 Terminals):**

**Terminal 1 (FastAPI Backend):**
```cmd
cd RAG_Excel
.venv\Scripts\activate
python main.py
```
*Keep this terminal running - this starts the API server*

**Terminal 2 (Web Interface):**
```cmd
cd RAG_Excel
.venv\Scripts\activate
streamlit run streamlit_app.py
```
*Open a new terminal window for this*

**Access URLs:**
- API: http://localhost:8000
- Web UI: http://localhost:8501
- API Docs: http://localhost:8000/docs

### **Method 2: Docker Deployment**

**Prerequisites for Docker:**
```cmd
# REQUIRED: Create .env file with your HuggingFace token
echo HUGGINGFACE_TOKEN=your_token_here > .env

# Add other required environment variables
echo MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0 >> .env
echo EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 >> .env
echo DEVICE=cpu >> .env
echo MAX_TOKENS=150 >> .env
```

**🔑 Get your free HuggingFace token:**
1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token (read access is sufficient)
3. Replace `your_token` with your actual token

#### **Option A: Quick Start (Recommended)**
```cmd
# API Only
docker-compose up --build rag-excel

# API + Web Interface
docker-compose --profile frontend up --build
```

#### **Option B: Manual Docker Build**
```cmd
# Build the Docker image
docker build -t rag-excel .

# Run API only
docker run -p 8000:8000 \
  -v ${PWD}/vector_store:/app/vector_store \
  -v ${PWD}/sample_files:/app/sample_files \
  --env-file .env \
  rag-excel

# Run with Streamlit frontend (requires 2 containers)
# Terminal 1 - API
docker run -p 8000:8000 --name rag-api \
  -v ${PWD}/vector_store:/app/vector_store \
  -v ${PWD}/sample_files:/app/sample_files \
  --env-file .env \
  rag-excel

# Terminal 2 - Streamlit
docker run -p 8501:8501 --link rag-api \
  -e API_BASE_URL=http://rag-api:8000 \
  rag-excel streamlit run streamlit_app.py --server.address 0.0.0.0
```

#### **Access URLs (Docker):**
- **API**: http://localhost:8000
- **Web UI**: http://localhost:8501 (if using frontend profile)
- **API Docs**: http://localhost:8000/docs

## 🧪 Test Your Installation

**Terminal 3 (Testing):**
```cmd
cd RAG_Excel
.venv\Scripts\activate
# Make sure main.py is running in Terminal 1, then:
python test_api.py

# Test enhanced document processing features:
python test_enhanced_ingestion.py
```


## 📖 How to Use

### **1. Upload Documents**
- **Web Interface**: Drag & drop files in Streamlit
- **API**: `POST /upload` with file attachment
- **Supported**: PDF, DOCX, TXT, CSV, DB, JPG, PNG

### **2. Ask Questions**
- **Web Interface**: Type questions in the text box
- **API**: `POST /query` with JSON `{"question": "your question"}`

### **3. Get Intelligent Answers**
The system will:
1. Find relevant content in your documents
2. Generate contextual answers using AI
3. Provide source attribution and confidence scores

##  API-Only Usage (Backend Without Frontend)

If you want to use only the RAG API without the Streamlit web interface (for integration into your own applications):

### **Step 1: Start Only the Backend**

```cmd
cd RAG_Excel
.venv\Scripts\activate

# Start only the FastAPI backend
python main.py
```

**Backend will be available at:** http://localhost:8000

### **Step 2: API Documentation**

Visit the interactive API docs:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### **Step 3: Basic API Workflow**

#### **1. Check System Health**
```bash
curl -X GET "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-01T10:30:00",
  "model_loaded": true,
  "vector_store_ready": true
}
```

#### **2. Upload Documents**
```bash
# Upload a PDF file
curl -X POST "http://localhost:8000/upload" \
  -F "file=@path/to/your/document.pdf"

# Upload multiple files
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document1.pdf"
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document2.docx"
```

**Response:**
```json
{
  "message": "Document processed successfully",
  "file_name": "document.pdf",
  "chunks_created": 15,
  "processing_time": 3.2
}
```

#### **3. Query Documents**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings in the report?"}'
```

**Response:**
```json
{
  "answer": "Based on the uploaded documents, the main findings include...",
  "source_files": ["document.pdf", "report.docx"],
  "confidence_score": 0.85,
  "retrieved_chunks": 5
}
```

#### **4. Get Vector Store Statistics**
```bash
curl -X GET "http://localhost:8000/stats"
```

**Response:**
```json
{
  "total_chunks": 150,
  "total_vectors": 150,
  "total_documents": 10,
  "embedding_dimension": 384,
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "vector_store_size": 5242880,
  "last_updated": "2025-08-01T10:25:00"
}
```

#### **5. Clear All Documents**
```bash
curl -X DELETE "http://localhost:8000/clear"
```

### **Step 4: Integration Examples**

#### **Python Integration**
```python
import requests
import json

# Initialize API client
API_BASE = "http://localhost:8000"

# Upload document
def upload_document(file_path):
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_BASE}/upload", files=files)
    return response.json()

# Query documents
def query_documents(question):
    data = {"question": question}
    response = requests.post(
        f"{API_BASE}/query", 
        headers={'Content-Type': 'application/json'},
        data=json.dumps(data)
    )
    return response.json()

# Example usage
result = upload_document("my_document.pdf")
print(f"Uploaded: {result['chunks_created']} chunks created")

answer = query_documents("What is this document about?")
print(f"Answer: {answer['answer']}")
print(f"Sources: {answer['source_files']}")
```

#### **JavaScript/Node.js Integration**
```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const API_BASE = 'http://localhost:8000';

// Upload document
async function uploadDocument(filePath) {
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));
    
    const response = await axios.post(`${API_BASE}/upload`, form, {
        headers: form.getHeaders()
    });
    return response.data;
}

// Query documents
async function queryDocuments(question) {
    const response = await axios.post(`${API_BASE}/query`, {
        question: question
    });
    return response.data;
}

// Example usage
async function main() {
    const uploadResult = await uploadDocument('./document.pdf');
    console.log(`Uploaded: ${uploadResult.chunks_created} chunks`);
    
    const queryResult = await queryDocuments('Summarize the key points');
    console.log(`Answer: ${queryResult.answer}`);
}
```

#### **cURL Batch Processing**
```bash
#!/bin/bash
# batch_upload.sh - Upload multiple documents

API_BASE="http://localhost:8000"

# Upload all PDFs in directory
for file in *.pdf; do
    echo "Uploading $file..."
    curl -X POST "$API_BASE/upload" -F "file=@$file"
    echo ""
done

# Query after all uploads
echo "Querying documents..."
curl -X POST "$API_BASE/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Provide a summary of all uploaded documents"}'
```

### **Step 5: Advanced API Features**

#### **Filtered Search (Coming Soon)**
```bash
# Search only in specific document types
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the sales figures?",
    "filters": {"source_type": "pdf"}
  }'
```

#### **Batch Queries**
```bash
# Process multiple questions at once
curl -X POST "http://localhost:8000/batch_query" \
  -H "Content-Type: application/json" \
  -d '{
    "questions": [
      "What is the main topic?",
      "Who are the key stakeholders?",
      "What are the recommendations?"
    ]
  }'
```

### **Step 6: Production Deployment**

For production API deployment:

1. **Use Production WSGI Server:**
   ```bash
   pip install gunicorn
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

2. **Configure Environment:**
   ```bash
   export DEVICE=cuda
   export MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
   export CHUNK_SIZE=512
   export TOP_K_RESULTS=5
   ```

3. **Docker API-Only Deployment:**
   ```bash
   docker-compose up --build rag-excel
   # API available at http://localhost:8000
   ```

## 🔧 API Reference

### **Core Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health status |
| `/upload` | POST | Upload documents |
| `/query` | POST | Ask questions |
| `/stats` | GET | Vector store statistics |
| `/clear` | DELETE | Clear all documents |

### **Example Usage**

**Upload a document:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

**Ask a question:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main points?"}'
```

## ⚙️ Configuration

### **Environment Variables**
Create `.env` file for custom settings:
```env
# AI Models (Get your free token from https://huggingface.co/settings/tokens)
HUGGINGFACE_TOKEN="hf_your_token_here"
MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Processing Settings
CHUNK_SIZE=512
CHUNK_OVERLAP=100
TOP_K_RESULTS=5
MAX_TOKENS=150

# Hardware
DEVICE=cuda  # or 'cpu' for CPU-only
VECTOR_STORE_PATH=./vector_store
```

### **Hardware Requirements**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8GB | 16GB+ |
| **GPU** | Any CUDA GPU | RTX 2060+ |
| **Storage** | 5GB | 10GB+ |
| **Python** | 3.9+ | 3.12+ |

## 🐳 Docker Information

The project includes a simplified Docker setup:

### **What's Included:**
- **Single Dockerfile**: CPU-optimized for reliable deployment
- **docker-compose.yml**: Multi-service orchestration
- **Production ready**: Python 3.12, proper health checks

### **Why CPU Docker?**
- ✅ **Simple deployment** - works on any server
- ✅ **No GPU dependencies** - fewer complications
- ✅ **Cost effective** - cheaper cloud hosting
- ✅ **Reliable** - consistent performance

### **For GPU Performance:**
Use local development setup with your RTX 2060 for best performance during development, then deploy with Docker for production.

## 🛠️ Troubleshooting

### **Common Issues & Solutions**

**❌ "Cannot connect to API"**
```cmd
# Check if server is running
curl http://localhost:8000/health
```

**❌ "Tesseract not found"**
```cmd
# Windows: Add to PATH
set PATH=%PATH%;C:\Program Files\Tesseract-OCR
```

**❌ "CUDA out of memory"**
```env
# Use CPU fallback in .env
DEVICE=cpu
```

**❌ "PyTorch not using GPU"**
```cmd
# Reinstall PyTorch with CUDA (REQUIRED for GPU)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU detection
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**❌ "Model download fails"**
```cmd
# Check internet connection and retry
python install.py
```

**❌ "Slow performance"**
- Use GPU: Set `DEVICE=cuda`
- Check GPU usage: `nvidia-smi`
- Reduce chunk size: `CHUNK_SIZE=256`

## 📁 Project Structure

```
RAG_Excel/
├── main.py                      # FastAPI backend server
├── streamlit_app.py            # Web interface
├── run.py                      # Automatic launcher (starts both backend & frontend)
├── ingestion.py                # Enhanced document processing with semantic chunking
├── embeddings.py               # Vector embeddings & FAISS
├── query.py                    # RAG engine & LLM
├── utils.py                    # OCR utilities
├── config.py                   # Configuration management
├── install.py                  # Automated installation script
├── test_api.py                 # API test suite
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Multi-service setup
├── README.md                   # This guide
```


## 🔄 Advanced Usage

### **Enhanced Document Processing**
The system now features **semantic chunking** that preserves document structure:
```python
# Automatically detects and preserves:
# - Paragraph boundaries
# - Table structures  
# - Page numbers
# - Content types (text, tables, figures)
```

### **Table-Aware Querying**
Ask specific questions about tables in your documents:
```
"What data is in the sales table?"
"Show me the pricing information from the product table"
```

### **Multiple Documents**
Upload multiple files - the system will search across all documents:
```python
# The system automatically combines context from multiple sources
```

### **Image OCR**
Upload images with text - OCR will extract and make it searchable:
```python
# Supports JPG, PNG with automatic text extraction
```

### **Custom Models**
Modify `config.py` to use different AI models:
```python
# Try different embedding models for better accuracy
# Or different LLMs for various response styles
```

---

## 🎉 Quick Summary - Start to Finish

### **🖥️ Local Development (Best Performance):**

1. **Clone & Setup Environment:**
   ```cmd
   git clone https://github.com/Radib-BK/RAG_Excel.git
   cd RAG_Excel
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **⚠️ CRITICAL: Create .env file**
   ```cmd
   # Create .env file
   notepad .env
   
   # Add this content (replace with your HuggingFace token):
   HUGGINGFACE_TOKEN=your_token_here
   MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   VECTOR_STORE_PATH=./vector_store
   CHUNK_SIZE=512
   CHUNK_OVERLAP=100
   TOP_K_RESULTS=5
   MAX_TOKENS=150
   DEVICE=cuda
   ```

3. **Install Dependencies:**
   
   **For GPU (RTX 2060+):**
   ```cmd
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   python install.py
   ```
   
   **For CPU Only:**
   ```cmd
   pip install -r requirements.txt
   python install.py
   ```

4. **Install Tesseract OCR:**
   - Download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Add `C:\Program Files\Tesseract-OCR` to PATH

5. **Start the Application:**
   ```cmd
   python run.py
   ```

6. **Access:** http://localhost:8501

### **🐳 Docker Deployment (Easy Setup):**

1. **Clone & Setup:**
   ```cmd
   git clone https://github.com/Radib-BK/RAG_Excel.git
   cd RAG_Excel
   
   # CRITICAL: Create .env file first
   echo HUGGINGFACE_TOKEN=your_token_here > .env
   echo MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0 >> .env
   echo EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 >> .env
   echo DEVICE=cpu >> .env
   ```

2. **Run with Docker:**
   ```cmd
   # API + Web Interface
   docker-compose --profile frontend up --build
   
   # OR API Only
   docker-compose up --build rag-excel
   ```

3. **Access:** http://localhost:8501 (frontend) or http://localhost:8000 (API only)

### **🧪 Test Installation:**
```cmd
# In a new terminal
python test_api.py
```

**Start by running `python run.py` and then access http://localhost:8501 to upload your first document!**

