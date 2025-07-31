# 🐳 Docker Deployment Guide

## 📋 **Simplified Docker Setup**

I've simplified the Docker configuration to avoid confusion. Now there's just **one Dockerfile** that's optimized for CPU deployment, which is the most practical approach for Docker containers.

### **Why CPU-only Docker?**
- ✅ **Simple setup** - no special requirements
- ✅ **Portable** - runs on any Docker host
- ✅ **Reliable** - fewer dependencies
- ✅ **Cost-effective** - cheaper cloud deployment
- ✅ **Python 3.12** - matches your local environment

## 🚀 **Quick Start Commands**

### **Option A: API Only**
```bash
# Build and run just the API
docker-compose up --build rag-chatbot

# Access API: http://localhost:8000
# API docs: http://localhost:8000/docs
```

### **Option B: API + Web Interface**
```bash
# Build and run API + Streamlit frontend
docker-compose --profile frontend up --build

# Access API: http://localhost:8000
# Access Web UI: http://localhost:8501
```

### **Option C: Manual Docker Build**
```bash
# Build image
docker build -t rag-chatbot .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/vector_store:/app/vector_store \
  -v $(pwd)/sample_files:/app/sample_files \
  rag-chatbot
```

## 📊 **What's Included**

| Component | Description | Port |
|-----------|-------------|------|
| **Dockerfile** | Single, simple CPU-optimized image | - |
| **rag-chatbot** | FastAPI backend service | 8000 |
| **streamlit-frontend** | Web interface (optional) | 8501 |
| **docker-compose.yml** | Multi-service orchestration | - |

## 🎯 **Deployment Recommendations**

### **For Development:**
- Use **local Python environment** with GPU (best performance)

### **For Production:**
- Use **Docker CPU version** (simple, reliable, portable)

### **For Sharing/Demo:**
- Use **docker-compose with frontend** (complete experience)

## 🔧 **Configuration**

All settings are controlled via environment variables:

```env
# Model Configuration
MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Performance
DEVICE=cpu
TORCH_DTYPE=float32
CHUNK_SIZE=512
MAX_TOKENS=150

# Storage
VECTOR_STORE_PATH=/app/vector_store
```

## 🧪 **Testing Docker Deployment**

```bash
# 1. Start services
docker-compose up rag-chatbot

# 2. Test health
curl http://localhost:8000/health

# 3. Upload test document
curl -X POST -F "file=@test.txt" http://localhost:8000/upload

# 4. Query
curl -X POST -H "Content-Type: application/json" \
  -d '{"question":"What is this about?"}' \
  http://localhost:8000/query
```

## 💡 **Best Practice**

**Use local development environment** for your RTX 2060 GPU, and **deploy with Docker** for production. This gives you:

- 🚀 **Best performance** during development
- 🔒 **Reliable deployment** for production
- 🎯 **Simple setup** without GPU complexity in containers

The single Dockerfile approach is much cleaner and easier to maintain! 🚀
