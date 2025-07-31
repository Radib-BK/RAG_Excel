# 🤖 RAG Chatbot - Complete Document Q&A System

A powerful Retrieval-Augmented Generation (RAG) chatbot that can process multiple document types, extract text using OCR, and answer questions using locally hosted AI models. Optimized for RTX 2060 GPUs but works on any hardware.

![RAG Chatbot Demo](screenshot/rag-page.png)

## ✨ Key Features

- 📄 **Multi-format Support**: PDF, DOCX, TXT, CSV, SQLite DB, JPG, PNG
- 🔍 **OCR Text Extraction**: Extract text from images using Tesseract
- 🧠 **Local AI Models**: TinyLlama LLM + SentenceTransformers embeddings
- ⚡ **GPU Optimized**: RTX 2060 ready with 4-bit quantization
- 🌐 **Web Interface**: FastAPI backend + Streamlit frontend
- 🐳 **Docker Ready**: Simple containerized deployment
- 💾 **Persistent Storage**: FAISS vector database for fast retrieval

## 🚀 Quick Start Guide

### **Method 1: Local Development (Recommended)**

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

#### **Step 2: Install Dependencies**

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

#### **Step 3: Install Tesseract OCR**
- **Windows**: Download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **Add to PATH**: `C:\Program Files\Tesseract-OCR`

#### **Step 4: Start the Application**

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
# Create .env file with your HuggingFace token
echo HUGGINGFACE_TOKEN=your_token_here > .env
```

#### **Option A: API Only**
```cmd
docker-compose up --build rag-chatbot
```

#### **Option B: API + Web Interface**
```cmd
docker-compose --profile frontend up --build
```

## 🧪 Test Your Installation

**Terminal 3 (Testing):**
```cmd
cd RAG_Excel
.venv\Scripts\activate
# Make sure main.py is running in Terminal 1, then:
python test_api.py
```

Expected output:
```
🧪 Starting RAG Chatbot API Tests
✅ API is healthy
✅ Upload successful - Chunks created: 2
✅ Query successful - Answer: This is a Retrieval-Augmented Generation...
🎉 All tests completed!
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
├── main.py              # FastAPI backend server
├── streamlit_app.py     # Web interface
├── ingestion.py         # Document processing
├── embeddings.py        # Vector embeddings & FAISS
├── query.py            # RAG engine & LLM
├── utils.py            # OCR utilities
├── config.py           # Configuration management
├── install.py          # Automated installation script
├── test_api.py         # Test suite
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Multi-service setup
└── README.md           # This guide
```


## 🔄 Advanced Usage

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

**Complete setup in 5 minutes:**

1. **Clone & Setup Environment:**
   ```cmd
   git clone https://github.com/Radib-BK/RAG_Excel.git
   cd RAG_Excel
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install Dependencies:**
   
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

3. **Install Tesseract OCR:**
   - Download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Add `C:\Program Files\Tesseract-OCR` to PATH

4. **Start the Application (2 Terminals):**
   
   **Terminal 1 - API Server:**
   ```cmd
   cd RAG_Excel
   .venv\Scripts\activate
   python main.py
   ```
   
   **Terminal 2 - Web Interface:**
   ```cmd
   cd RAG_Excel  
   .venv\Scripts\activate
   streamlit run streamlit_app.py
   ```

5. **Access & Test:**
   - Web UI: http://localhost:8501
   - API: http://localhost:8000
   - Test: `python test_api.py` (in Terminal 3)

**Start by running `python main.py` and uploading your first document!** 🚀

