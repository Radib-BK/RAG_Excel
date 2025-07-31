"""
Embeddings and vector store management module
Handles text embedding generation and FAISS vector store operations with production-grade features
"""

import os
import json
import pickle
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
import numpy as np
from datetime import datetime
from hashlib import sha256

# ML imports
from sentence_transformers import SentenceTransformer
import faiss

from utils import clean_text

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Advanced embeddings and vector store management with production features"""
    
    def __init__(self):
        self.model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        self.vector_store_path = os.getenv('VECTOR_STORE_PATH', './vector_store')
        self.device = 'cuda' if os.getenv('DEVICE', 'cuda') == 'cuda' else 'cpu'
        self.vector_metric = os.getenv('VECTOR_METRIC', 'cosine')  # 'cosine' or 'l2'
        self.use_memory_mapping = os.getenv('USE_MEMORY_MAPPING', 'false').lower() == 'true'
        
        # HNSW parameters for scalable search
        self.hnsw_m = int(os.getenv('HNSW_M', '32'))  # Number of bi-directional links
        self.hnsw_ef_search = int(os.getenv('HNSW_EF_SEARCH', '64'))  # Search quality
        
        # Initialize components
        self.model = None
        self.index = None
        self.metadata = []
        self.dimension = None
        
        # Embedding cache for performance
        self.embedding_cache = {}
        self.cache_file = os.path.join(self.vector_store_path, 'embedding_cache.pkl')
        
        # Create vector store directory
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Load caches and vector store
        self._load_embedding_cache()
        self._load_vector_store()
        
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return sha256(text.encode('utf-8')).hexdigest()
        
    def _load_embedding_cache(self):
        """Load embedding cache from disk"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
        except Exception as e:
            logger.warning(f"Could not load embedding cache: {e}")
            self.embedding_cache = {}
            
    def _save_embedding_cache(self):
        """Save embedding cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.debug(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Could not save embedding cache: {e}")
    
    def _load_embedding_model(self):
        """Load the sentence transformer model with auto-dimension detection"""
        if self.model is None:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name, device=self.device)
                
                # Auto-detect embedding dimension
                self.dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")
                
                # Update metadata with model version
                self.model_version = self.model_name
                
            except Exception as e:
                logger.error(f"Error loading embedding model: {str(e)}")
                raise
    
    def _initialize_faiss_index(self):
        """Initialize FAISS index with configurable distance metric and HNSW for scalability"""
        if self.index is None:
            if self.dimension is None:
                self._load_embedding_model()
                
            # Choose index type based on metric and scalability needs
            if self.vector_metric == 'cosine':
                # HNSW for scalable approximate search with cosine similarity
                self.index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m)
                self.index.hnsw.efSearch = self.hnsw_ef_search
                logger.info(f"Initialized HNSW index for cosine similarity (M={self.hnsw_m}, efSearch={self.hnsw_ef_search})")
            elif self.vector_metric == 'l2':
                # HNSW with L2 distance
                self.index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m, faiss.METRIC_L2)
                self.index.hnsw.efSearch = self.hnsw_ef_search
                logger.info(f"Initialized HNSW index for L2 distance (M={self.hnsw_m}, efSearch={self.hnsw_ef_search})")
            else:
                # Fallback to exact search
                if self.vector_metric == 'cosine':
                    self.index = faiss.IndexFlatIP(self.dimension)
                else:
                    self.index = faiss.IndexFlatL2(self.dimension)
                logger.info(f"Initialized exact search index with {self.vector_metric} metric")
    
    def _load_vector_store(self):
        """Load existing vector store from disk with memory mapping option"""
        index_path = os.path.join(self.vector_store_path, 'faiss_index.idx')
        metadata_path = os.path.join(self.vector_store_path, 'faiss_index.metadata.json')
        # Backward compatibility
        old_metadata_path = os.path.join(self.vector_store_path, 'metadata.json')
        
        try:
            if os.path.exists(index_path) and (os.path.exists(metadata_path) or os.path.exists(old_metadata_path)):
                # Load FAISS index with optional memory mapping
                if self.use_memory_mapping:
                    self.index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
                    logger.info("Loaded FAISS index with memory mapping")
                else:
                    self.index = faiss.read_index(index_path)
                    logger.info("Loaded FAISS index into memory")
                
                logger.info(f"Index contains {self.index.ntotal} vectors")
                
                # Load metadata (try new format first, fallback to old)
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'metadata' in data:
                            # New versioned format
                            self.metadata = data['metadata']
                            logger.info(f"Loaded versioned metadata with {data.get('index_info', {}).get('total_vectors', 'unknown')} vectors")
                        else:
                            # Old format
                            self.metadata = data
                elif os.path.exists(old_metadata_path):
                    with open(old_metadata_path, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)
                    logger.info("Loaded legacy metadata format")
                
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
                metadata_path = os.path.join(self.vector_store_path, 'faiss_index.metadata.json')
                
                # Save FAISS index
                faiss.write_index(self.index, index_path)
                
                # Save metadata with versioning info
                metadata_with_version = {
                    'metadata': self.metadata,
                    'index_info': {
                        'faiss_version': faiss.__version__ if hasattr(faiss, '__version__') else 'unknown',
                        'model_version': getattr(self, 'model_version', self.model_name),
                        'vector_metric': self.vector_metric,
                        'dimension': self.dimension,
                        'total_vectors': self.index.ntotal,
                        'last_updated': datetime.now().isoformat()
                    }
                }
                
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata_with_version, f, indent=2, ensure_ascii=False)
                
                # Save embedding cache
                self._save_embedding_cache()
                
                logger.info("Vector store saved successfully")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts with caching support
        
        Args:
            texts: List of text strings
            use_cache: Whether to use embedding cache
            
        Returns:
            Numpy array of embeddings
        """
        if not self.model:
            self._load_embedding_model()
        
        try:
            # Clean texts
            cleaned_texts = [clean_text(text) for text in texts]
            embeddings = []
            texts_to_embed = []
            cache_keys = []
            
            # Check cache for existing embeddings
            for text in cleaned_texts:
                cache_key = self._get_cache_key(text)
                cache_keys.append(cache_key)
                
                if use_cache and cache_key in self.embedding_cache:
                    embeddings.append(self.embedding_cache[cache_key])
                else:
                    embeddings.append(None)  # Placeholder
                    texts_to_embed.append(text)
            
            # Generate embeddings for uncached texts
            if texts_to_embed:
                logger.info(f"Generating embeddings for {len(texts_to_embed)} new texts")
                new_embeddings = self.model.encode(
                    texts_to_embed,
                    convert_to_numpy=True,
                    normalize_embeddings=(self.vector_metric == 'cosine'),
                    show_progress_bar=len(texts_to_embed) > 10
                )
                
                # Cache new embeddings and fill placeholders
                new_emb_idx = 0
                for i, embedding in enumerate(embeddings):
                    if embedding is None:  # This was a placeholder
                        new_emb = new_embeddings[new_emb_idx]
                        embeddings[i] = new_emb
                        
                        # Cache it
                        if use_cache:
                            self.embedding_cache[cache_keys[i]] = new_emb
                        
                        new_emb_idx += 1
            
            result = np.array(embeddings)
            logger.info(f"Generated/retrieved embeddings for {len(texts)} texts ({len(texts_to_embed)} new, {len(texts) - len(texts_to_embed)} cached)")
            return result
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    async def generate_embeddings_async(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """
        Asynchronous version of embedding generation for non-blocking operations
        
        Args:
            texts: List of text strings
            use_cache: Whether to use embedding cache
            
        Returns:
            Numpy array of embeddings
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_embeddings, texts, use_cache)
    
    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        Add document chunks to the vector store with enhanced metadata
        
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
            
            # Generate embeddings with caching
            embeddings = self.generate_embeddings(texts, use_cache=True)
            
            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))
            
            # Add enhanced metadata
            for i, chunk in enumerate(chunks):
                metadata_entry = {
                    # Original chunk data
                    'text': chunk['text'],
                    'source': chunk.get('source', 'Unknown'),
                    'source_type': chunk.get('source_type', 'unknown'),
                    'chunk_index': chunk.get('chunk_index', 0),
                    'word_count': chunk.get('word_count', 0),
                    'char_count': chunk.get('char_count', 0),
                    'estimated_tokens': chunk.get('estimated_tokens', 0),
                    
                    # Enhanced metadata
                    'doc_id': chunk.get('doc_id'),
                    'page': chunk.get('page'),
                    'has_table': chunk.get('has_table', False),
                    'has_figure': chunk.get('has_figure', False),
                    
                    # System metadata
                    'timestamp': datetime.now().isoformat(),
                    'vector_id': len(self.metadata),
                    'model_version': getattr(self, 'model_version', self.model_name),
                    'embedding_dimension': self.dimension
                }
                self.metadata.append(metadata_entry)
            
            # Save to disk
            self._save_vector_store()
            
            logger.info(f"Added {len(chunks)} chunks to vector store. Total vectors: {self.index.ntotal}")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def search(self, query: str, top_k: int = None, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Search for similar documents with enhanced output format
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters for metadata (e.g., {'source_type': 'pdf'})
            
        Returns:
            Structured search results with metadata and timing
        """
        if top_k is None:
            top_k = int(os.getenv('TOP_K_RESULTS', 5))
        
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Vector store is empty. Ingest documents first.")
        
        if not self.model:
            self._load_embedding_model()
        
        search_start = datetime.now()
        
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query], use_cache=True)
            
            # Search in FAISS (get more results for filtering if needed)
            search_k = top_k * 3 if filters else top_k
            scores, indices = self.index.search(query_embedding.astype('float32'), search_k)
            
            # Prepare results with filtering
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0 and idx < len(self.metadata):  # Valid index
                    result = self.metadata[idx].copy()
                    
                    # Convert FAISS scores to proper similarity scores
                    if self.vector_metric == 'cosine':
                        # For HNSW with cosine (IP), score is already similarity (0-1)
                        # But we need to ensure it's in proper range
                        similarity_score = max(0.0, min(1.0, float(score)))
                    else:  # L2 distance
                        # Convert L2 distance to similarity (higher distance = lower similarity)
                        # Use exponential decay: similarity = exp(-distance/scale)
                        similarity_score = max(0.0, min(1.0, 1.0 / (1.0 + float(score))))
                    
                    result['similarity_score'] = similarity_score
                    result['rank'] = len(results) + 1
                    
                    # Apply filters if provided
                    if filters:
                        skip = False
                        for filter_key, filter_value in filters.items():
                            if result.get(filter_key) != filter_value:
                                skip = True
                                break
                        if skip:
                            continue
                    
                    results.append(result)
                    
                    # Stop if we have enough results
                    if len(results) >= top_k:
                        break
            
            search_end = datetime.now()
            search_time = (search_end - search_start).total_seconds()
            
            # Return structured response
            response = {
                "query": query,
                "top_k": top_k,
                "filters": filters,
                "total_found": len(results),
                "search_time_seconds": search_time,
                "results": results,
                "retrieved_at": search_end.isoformat(),
                "vector_store_stats": {
                    "total_vectors": self.index.ntotal,
                    "embedding_dimension": self.dimension,
                    "index_type": type(self.index).__name__
                }
            }
            
            logger.info(f"Found {len(results)} results for query in {search_time:.3f}s")
            return response
            
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
        """Clear the vector store and caches"""
        try:
            # Remove files
            index_path = os.path.join(self.vector_store_path, 'faiss_index.idx')
            metadata_path = os.path.join(self.vector_store_path, 'faiss_index.metadata.json')
            old_metadata_path = os.path.join(self.vector_store_path, 'metadata.json')  # Legacy
            cache_path = self.cache_file
            
            for file_path in [index_path, metadata_path, old_metadata_path, cache_path]:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Removed {file_path}")
            
            # Reset in-memory structures
            self.index = None
            self.metadata = []
            self.embedding_cache = {}
            
            logger.info("Vector store and caches cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise
