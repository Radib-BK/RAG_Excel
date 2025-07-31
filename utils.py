"""
Utility functions for the RAG chatbot
Includes OCR, logging setup, environment loading, and helper functions
"""

import os
import logging
import base64
from io import BytesIO
from typing import Optional
import pytesseract
from PIL import Image
from dotenv import load_dotenv

def load_environment_variables():
    """Load environment variables from .env file"""
    load_dotenv()
    
    # Set default values if not provided
    os.environ.setdefault('CHUNK_SIZE', '512')
    os.environ.setdefault('CHUNK_OVERLAP', '100')
    os.environ.setdefault('TOP_K_RESULTS', '5')
    os.environ.setdefault('MAX_TOKENS', '150')
    os.environ.setdefault('DEVICE', 'cuda')
    os.environ.setdefault('VECTOR_STORE_PATH', './vector_store')
    os.environ.setdefault('MODEL_NAME', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    os.environ.setdefault('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('rag_chatbot.log'),
            logging.StreamHandler()
        ]
    )

def extract_text_from_image_base64(image_base64: str) -> str:
    """
    Extract text from base64 encoded image using OCR
    
    Args:
        image_base64: Base64 encoded image string
        
    Returns:
        Extracted text from the image
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract text using pytesseract
        text = pytesseract.image_to_string(image, lang='eng')
        
        return text.strip()
        
    except Exception as e:
        logging.error(f"Error extracting text from image: {str(e)}")
        return ""

def extract_text_from_image_file(image_path: str) -> str:
    """
    Extract text from image file using OCR
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Extracted text from the image
    """
    try:
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract text using pytesseract
        text = pytesseract.image_to_string(image, lang='eng')
        
        return text.strip()
        
    except Exception as e:
        logging.error(f"Error extracting text from image file {image_path}: {str(e)}")
        return ""

def validate_file_size(file_path: str, max_size_mb: int = 50) -> bool:
    """
    Validate file size
    
    Args:
        file_path: Path to the file
        max_size_mb: Maximum allowed file size in MB
        
    Returns:
        True if file size is within limit, False otherwise
    """
    try:
        file_size = os.path.getsize(file_path)
        max_size_bytes = max_size_mb * 1024 * 1024
        return file_size <= max_size_bytes
    except Exception:
        return False

def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove special characters that might interfere with processing
    # Keep basic punctuation and alphanumeric characters
    import re
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'\/]', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension (lowercase, without dot)
    """
    return os.path.splitext(filename)[1][1:].lower()

def format_chunks_for_context(chunks: list, max_length: int = 2000) -> str:
    """
    Format retrieved chunks into context for LLM
    
    Args:
        chunks: List of document chunks
        max_length: Maximum length of formatted context
        
    Returns:
        Formatted context string
    """
    context_parts = []
    current_length = 0
    
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.get('text', '')
        source = chunk.get('source', 'Unknown')
        
        formatted_chunk = f"[Source: {source}]\n{chunk_text}\n"
        
        if current_length + len(formatted_chunk) > max_length:
            break
            
        context_parts.append(formatted_chunk)
        current_length += len(formatted_chunk)
    
    return "\n".join(context_parts)

def create_rag_prompt(question: str, context: str) -> str:
    """
    Create a prompt for the RAG system
    
    Args:
        question: User's question
        context: Retrieved context from documents
        
    Returns:
        Formatted prompt for the LLM
    """
    prompt = f"""You are a helpful assistant. Use the following context to answer the user's question accurately and concisely.

Instructions:
- If the question asks for categories, types, or unique values, provide only the distinct items without duplicates
- For table data, analyze the content carefully to extract the specific information requested
- If asking for unique categories from a table, list only the category names (e.g., "Fruits, Bakery, Dairy") not item-category pairs
- Be precise and direct in your answer
- If the answer cannot be found in the context, say "I don't have enough information to answer this question based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""
    
    return prompt

def estimate_tokens(text: str) -> int:
    """
    Estimate number of tokens in text (rough approximation)
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    # Rough estimation: 1 token ≈ 4 characters
    return len(text) // 4

def truncate_text(text: str, max_tokens: int) -> str:
    """
    Truncate text to approximate token limit
    
    Args:
        text: Input text
        max_tokens: Maximum number of tokens
        
    Returns:
        Truncated text
    """
    max_chars = max_tokens * 4  # Rough approximation
    if len(text) <= max_chars:
        return text
    
    # Truncate at word boundary
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > 0:
        truncated = truncated[:last_space]
    
    return truncated + "..."
