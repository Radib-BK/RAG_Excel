"""
Embeddings and vector store management module
Handles text embedding generation and FAISS vector store operations
"""

import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime

# ML imports
from sentence_transformers import SentenceTransformer
import faiss

from utils import clean_text

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages text embeddings and FAISS vector store"""
    
    def __init__(self):
        self.model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        self.vector_store_path = os.getenv('VECTOR_STORE_PATH', './vector_store')
        self.device = 'cuda' if os.getenv('DEVICE', 'cuda') == 'cuda' else 'cpu'
        
        # Initialize components
        self.model = None
        self.index = None
        self.metadata = []
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        
        # Create vector store directory
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Load existing vector store if available
        self._load_vector_store()
    
    def _load_embedding_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")
            except Exception as e:
                logger.error(f"Error loading embedding model: {str(e)}")
                raise
    
    def _initialize_faiss_index(self):
        """Initialize FAISS index"""
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            logger.info("Initialized new FAISS index")
    
    def _load_vector_store(self):
        """Load existing vector store from disk"""
        index_path = os.path.join(self.vector_store_path, 'faiss_index.idx')
        metadata_path = os.path.join(self.vector_store_path, 'metadata.json')
        
        try:
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                # Load FAISS index
                self.index = faiss.read_index(index_path)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
                
                # Load metadata
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                
                self.dimension = self.index.d
                logger.info("Vector store loaded successfully")
            else:
                logger.info("No existing vector store found")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            self.index = None
            self.metadata = []
    
    def _save_vector_store(self):
        """Save vector store to disk"""
        try:
            if self.index is not None:
                index_path = os.path.join(self.vector_store_path, 'faiss_index.idx')
                metadata_path = os.path.join(self.vector_store_path, 'metadata.json')
                
                # Save FAISS index
                faiss.write_index(self.index, index_path)
                
                # Save metadata
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(self.metadata, f, indent=2, ensure_ascii=False)
                
                logger.info("Vector store saved successfully")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        if not self.model:
            self._load_embedding_model()
        
        try:
            # Clean texts
            cleaned_texts = [clean_text(text) for text in texts]
            
            # Generate embeddings
            embeddings = self.model.encode(
                cleaned_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,  # For cosine similarity
                show_progress_bar=len(texts) > 10
            )
            
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of document chunks with metadata
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        # Initialize if needed
        if not self.model:
            self._load_embedding_model()
        if self.index is None:
            self._initialize_faiss_index()
        
        try:
            # Extract texts for embedding
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))
            
            # Add metadata
            for i, chunk in enumerate(chunks):
                metadata_entry = {
                    'text': chunk['text'],
                    'source': chunk.get('source', 'Unknown'),
                    'chunk_index': chunk.get('chunk_index', 0),
                    'word_count': chunk.get('word_count', 0),
                    'char_count': chunk.get('char_count', 0),
                    'estimated_tokens': chunk.get('estimated_tokens', 0),
                    'timestamp': datetime.now().isoformat(),
                    'vector_id': len(self.metadata)
                }
                self.metadata.append(metadata_entry)
            
            # Save to disk
            self._save_vector_store()
            
            logger.info(f"Added {len(chunks)} chunks to vector store. Total vectors: {self.index.ntotal}")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results with metadata and scores
        """
        if top_k is None:
            top_k = int(os.getenv('TOP_K_RESULTS', 5))
        
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        if not self.model:
            self._load_embedding_model()
        
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])
            
            # Search in FAISS
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0 and idx < len(self.metadata):  # Valid index
                    result = self.metadata[idx].copy()
                    result['similarity_score'] = float(score)
                    result['rank'] = i + 1
                    results.append(result)
            
            logger.info(f"Found {len(results)} results for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise
    
    def vector_store_exists(self) -> bool:
        """Check if vector store exists and has data"""
        return self.index is not None and self.index.ntotal > 0
    
    def is_model_loaded(self) -> bool:
        """Check if embedding model is loaded"""
        return self.model is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        stats = {
            'total_chunks': len(self.metadata),
            'total_vectors': self.index.ntotal if self.index else 0,
            'embedding_dimension': self.dimension,
            'model_name': self.model_name,
            'vector_store_size': 0
        }
        
        # Calculate vector store size
        try:
            index_path = os.path.join(self.vector_store_path, 'faiss_index.idx')
            metadata_path = os.path.join(self.vector_store_path, 'metadata.json')
            
            if os.path.exists(index_path):
                stats['vector_store_size'] += os.path.getsize(index_path)
            if os.path.exists(metadata_path):
                stats['vector_store_size'] += os.path.getsize(metadata_path)
        except Exception:
            pass
        
        # Get unique source files
        sources = set()
        for item in self.metadata:
            sources.add(item.get('source', 'Unknown'))
        stats['total_documents'] = len(sources)
        
        # Get last updated time
        if self.metadata:
            timestamps = [item.get('timestamp', '') for item in self.metadata]
            timestamps = [t for t in timestamps if t]
            if timestamps:
                stats['last_updated'] = max(timestamps)
        
        return stats
    
    def clear_vector_store(self):
        """Clear the vector store"""
        try:
            # Remove files
            index_path = os.path.join(self.vector_store_path, 'faiss_index.idx')
            metadata_path = os.path.join(self.vector_store_path, 'metadata.json')
            
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            # Reset in-memory structures
            self.index = None
            self.metadata = []
            
            logger.info("Vector store cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise
