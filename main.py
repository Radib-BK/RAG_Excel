"""
FastAPI Main Application for RAG Chatbot
Handles file uploads and query endpoints
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import shutil
import tempfile
import time
from pathlib import Path
import logging

from ingestion import DocumentProcessor
from embeddings import EmbeddingManager
from query import RAGQueryEngine
from utils import setup_logging, load_environment_variables

# Load environment variables
load_environment_variables()

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="A Retrieval-Augmented Generation chatbot that processes documents and answers questions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
document_processor = DocumentProcessor()
embedding_manager = EmbeddingManager()
rag_engine = RAGQueryEngine(embedding_manager)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str
    image_base64: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    source_files: List[str]
    confidence_score: float
    processing_time: float

class UploadResponse(BaseModel):
    message: str
    file_name: str
    chunks_created: int
    processing_time: float

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "RAG Chatbot API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Check if embedding model is loaded
        model_status = embedding_manager.is_model_loaded()
        vector_store_status = embedding_manager.vector_store_exists()
        
        return {
            "status": "healthy",
            "embedding_model_loaded": model_status,
            "vector_store_exists": vector_store_status,
            "supported_formats": document_processor.supported_formats
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service unhealthy")

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document for RAG
    Supported formats: .pdf, .docx, .txt, .csv, .db, .jpg, .png
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Processing upload: {file.filename}")
        
        # Validate file format
        if not document_processor.is_supported_format(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported: {document_processor.supported_formats}"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_file_path = tmp_file.name
        
        try:
            # Process document using enhanced pipeline
            chunks = document_processor.process_document(tmp_file_path, file.filename)
            
            if not chunks:
                raise HTTPException(status_code=400, detail="No content could be extracted from the document")
            
            # Generate embeddings and store in vector database
            embedding_manager.add_documents(chunks)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Successfully processed {file.filename}: {len(chunks)} chunks created")
            
            return UploadResponse(
                message="Document processed successfully",
                file_name=file.filename,
                chunks_created=len(chunks),
                processing_time=processing_time
            )
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing upload {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the RAG system with a question
    Optionally include an image in base64 format for OCR
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: {request.question[:100]}...")
        
        # Check if vector store exists
        if not embedding_manager.vector_store_exists():
            raise HTTPException(
                status_code=400,
                detail="No documents have been uploaded yet. Please upload documents first."
            )
        
        # Process the query through RAG pipeline
        result = await rag_engine.query(
            question=request.question,
            image_base64=request.image_base64
        )
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            answer=result["answer"],
            source_files=result["source_files"],
            confidence_score=result["confidence_score"],
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get statistics about the vector store"""
    try:
        stats = embedding_manager.get_stats()
        return {
            "total_documents": stats.get("total_documents", 0),
            "total_chunks": stats.get("total_chunks", 0),
            "vector_store_size": stats.get("vector_store_size", 0),
            "last_updated": stats.get("last_updated", None)
        }
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving statistics")

@app.delete("/clear")
async def clear_vector_store():
    """Clear all documents from the vector store"""
    try:
        embedding_manager.clear_vector_store()
        return {"message": "Vector store cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing vector store: {str(e)}")
        raise HTTPException(status_code=500, detail="Error clearing vector store")

if __name__ == "__main__":
    import uvicorn
    
    # Create necessary directories
    os.makedirs("vector_store", exist_ok=True)
    os.makedirs("sample_files", exist_ok=True)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disabled auto-reload to prevent watchfiles issues
        log_level="info"
    )
