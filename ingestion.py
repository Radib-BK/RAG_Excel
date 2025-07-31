"""
Document ingestion and processing module
Handles text extraction from various file formats and text chunking
"""

import os
import logging
import sqlite3
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd

# Document processing imports
import pdfplumber
from docx import Document

from utils import (
    extract_text_from_image_file, 
    clean_text, 
    get_file_extension,
    estimate_tokens
)

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing and text extraction"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt', '.csv', '.db', '.jpg', '.jpeg', '.png']
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 512))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 100))
    
    def is_supported_format(self, filename: str) -> bool:
        """Check if file format is supported"""
        extension = get_file_extension(filename)
        return f'.{extension}' in self.supported_formats
    
    def extract_text(self, file_path: str, filename: str) -> str:
        """
        Extract text from various file formats
        
        Args:
            file_path: Path to the file
            filename: Original filename
            
        Returns:
            Extracted text content
        """
        extension = get_file_extension(filename)
        
        try:
            if extension == 'pdf':
                return self._extract_from_pdf(file_path)
            elif extension == 'docx':
                return self._extract_from_docx(file_path)
            elif extension == 'txt':
                return self._extract_from_txt(file_path)
            elif extension == 'csv':
                return self._extract_from_csv(file_path)
            elif extension == 'db':
                return self._extract_from_db(file_path)
            elif extension in ['jpg', 'jpeg', 'png']:
                return self._extract_from_image(file_path)
            else:
                raise ValueError(f"Unsupported file format: {extension}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {str(e)}")
            raise
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using pdfplumber"""
        text_content = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        # Add page marker for better source tracking
                        text_content.append(f"[Page {page_num + 1}]\n{page_text}")
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                    continue
        
        return "\n\n".join(text_content)
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        doc = Document(file_path)
        text_content = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content.append(paragraph.text)
        
        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    text_content.append(" | ".join(row_text))
        
        return "\n".join(text_content)
    
    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
        
        raise ValueError("Could not decode text file with any supported encoding")
    
    def _extract_from_csv(self, file_path: str) -> str:
        """Extract text from CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            # Convert DataFrame to readable text format
            text_content = []
            text_content.append(f"CSV Data with {len(df)} rows and {len(df.columns)} columns")
            text_content.append(f"Columns: {', '.join(df.columns.tolist())}")
            text_content.append("")
            
            # Add sample of data (first 100 rows to avoid huge chunks)
            sample_size = min(100, len(df))
            for idx, row in df.head(sample_size).iterrows():
                row_text = []
                for col, value in row.items():
                    if pd.notna(value):
                        row_text.append(f"{col}: {value}")
                text_content.append(" | ".join(row_text))
            
            if len(df) > 100:
                text_content.append(f"... and {len(df) - 100} more rows")
            
            return "\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            raise
    
    def _extract_from_db(self, file_path: str) -> str:
        """Extract text from SQLite database"""
        try:
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            text_content = []
            text_content.append(f"Database contains {len(tables)} tables")
            
            for table_name in tables:
                table_name = table_name[0]
                text_content.append(f"\nTable: {table_name}")
                
                # Get table schema
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                column_names = [col[1] for col in columns]
                text_content.append(f"Columns: {', '.join(column_names)}")
                
                # Get sample data (first 50 rows)
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 50")
                rows = cursor.fetchall()
                
                for row in rows:
                    row_text = []
                    for i, value in enumerate(row):
                        if value is not None:
                            row_text.append(f"{column_names[i]}: {value}")
                    text_content.append(" | ".join(row_text))
                
                # Get total count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                total_rows = cursor.fetchone()[0]
                if total_rows > 50:
                    text_content.append(f"... and {total_rows - 50} more rows in {table_name}")
            
            conn.close()
            return "\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Error reading database file: {str(e)}")
            raise
    
    def _extract_from_image(self, file_path: str) -> str:
        """Extract text from image using OCR"""
        return extract_text_from_image_file(file_path)
    
    def chunk_text(self, text: str, source_file: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks with overlap
        
        Args:
            text: Text to chunk
            source_file: Source file name for metadata
            
        Returns:
            List of chunk dictionaries with metadata
        """
        cleaned_text = clean_text(text)
        
        if not cleaned_text:
            return []
        
        # Simple token-based chunking (approximate)
        words = cleaned_text.split()
        chunks = []
        
        # Convert token sizes to approximate word counts
        chunk_size_words = self.chunk_size // 4  # Rough approximation
        overlap_words = self.chunk_overlap // 4
        
        for i in range(0, len(words), chunk_size_words - overlap_words):
            chunk_words = words[i:i + chunk_size_words]
            chunk_text = " ".join(chunk_words)
            
            # Skip very short chunks
            if len(chunk_words) < 10:
                continue
            
            chunk = {
                'text': chunk_text,
                'source': source_file,
                'chunk_index': len(chunks),
                'word_count': len(chunk_words),
                'char_count': len(chunk_text),
                'estimated_tokens': estimate_tokens(chunk_text)
            }
            
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from {source_file}")
        return chunks
    
    def process_document(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """
        Complete document processing pipeline
        
        Args:
            file_path: Path to the document file
            filename: Original filename
            
        Returns:
            List of processed chunks with metadata
        """
        logger.info(f"Processing document: {filename}")
        
        # Extract text
        extracted_text = self.extract_text(file_path, filename)
        
        if not extracted_text.strip():
            logger.warning(f"No text extracted from {filename}")
            return []
        
        logger.info(f"Extracted {len(extracted_text)} characters from {filename}")
        
        # Create chunks
        chunks = self.chunk_text(extracted_text, filename)
        
        return chunks
