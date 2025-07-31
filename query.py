"""
RAG Query Engine
Handles query processing, retrieval, and generation using LLM
"""

import os
import logging
import torch
from typing import Dict, Any, Optional, List
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)

from embeddings import EmbeddingManager
from utils import (
    extract_text_from_image_base64,
    create_rag_prompt,
    format_chunks_for_context,
    clean_text
)

logger = logging.getLogger(__name__)

class RAGQueryEngine:
    """Handles RAG query processing and generation"""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.model_name = os.getenv('MODEL_NAME', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
        self.device = 'cuda' if torch.cuda.is_available() and os.getenv('DEVICE', 'cuda') == 'cuda' else 'cpu'
        self.max_tokens = int(os.getenv('MAX_TOKENS', 150))
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.generator = None
        
        # Load model on initialization
        self._load_llm()
    
    def _load_llm(self):
        """Load the language model for generation"""
        try:
            logger.info(f"Loading LLM: {self.model_name} on device: {self.device}")
            
            # Configure for low memory usage
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == 'cuda' else torch.float32,
                "device_map": "auto" if self.device == 'cuda' else None,
                "trust_remote_code": True
            }
            
            # For GPU with limited VRAM, use quantization
            if self.device == 'cuda':
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = quantization_config
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Create text generation pipeline (let accelerate handle device placement)
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                # Don't specify device when using accelerate with device_map="auto"
                return_full_text=False,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=self.max_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("LLM loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading LLM: {str(e)}")
            # Fallback to CPU without quantization
            if self.device == 'cuda':
                logger.info("Falling back to CPU")
                self.device = 'cpu'
                self._load_llm_cpu_fallback()
            else:
                raise
    
    def _load_llm_cpu_fallback(self):
        """Fallback LLM loading for CPU"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,  # CPU
                return_full_text=False,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=self.max_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("LLM loaded successfully on CPU")
            
        except Exception as e:
            logger.error(f"Error loading LLM on CPU: {str(e)}")
            raise
    
    def _process_image_query(self, image_base64: str) -> str:
        """Process image input using OCR"""
        try:
            logger.info("Processing image with OCR")
            extracted_text = extract_text_from_image_base64(image_base64)
            
            if extracted_text.strip():
                logger.info(f"Extracted {len(extracted_text)} characters from image")
                return extracted_text
            else:
                logger.warning("No text extracted from image")
                return ""
                
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return ""
    
    def _retrieve_relevant_chunks(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for the query"""
        if top_k is None:
            top_k = int(os.getenv('TOP_K_RESULTS', 5))
        
        try:
            search_response = self.embedding_manager.search(query, top_k=top_k)
            # Extract results from the new structured response format
            results = search_response.get('results', []) if isinstance(search_response, dict) else search_response
            logger.info(f"Retrieved {len(results)} chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            return []
    
    def _generate_answer(self, prompt: str) -> str:
        """Generate answer using the LLM"""
        try:
            if not self.generator:
                raise ValueError("LLM not loaded")
            
            # Generate response
            outputs = self.generator(prompt)
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]['generated_text']
                # Clean up the generated text
                answer = clean_text(generated_text)
                return answer
            else:
                return "I apologize, but I couldn't generate a response."
                
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def _calculate_confidence_score(self, query: str, chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieval results"""
        if not chunks:
            return 0.0
        
        # Simple confidence calculation based on similarity scores
        scores = [chunk.get('similarity_score', 0.0) for chunk in chunks]
        
        if scores:
            # Average of top scores with some normalization
            avg_score = sum(scores) / len(scores)
            # Normalize to 0-1 range (assuming similarity scores are already normalized)
            confidence = min(1.0, max(0.0, avg_score))
            return confidence
        
        return 0.0
    
    async def query(self, question: str, image_base64: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline
        
        Args:
            question: User's question
            image_base64: Optional base64 encoded image for OCR
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # Prepare the complete query
            complete_query = question
            
            # Process image if provided
            if image_base64:
                image_text = self._process_image_query(image_base64)
                if image_text:
                    complete_query = f"{question}\n\nAdditional context from image: {image_text}"
            
            # Retrieve relevant chunks
            retrieved_chunks = self._retrieve_relevant_chunks(complete_query)
            
            if not retrieved_chunks:
                return {
                    "answer": "I don't have any relevant information in my knowledge base to answer this question. Please upload some documents first.",
                    "source_files": [],
                    "confidence_score": 0.0
                }
            
            # Format context for the LLM
            context = format_chunks_for_context(retrieved_chunks, max_length=1500)
            
            # Create RAG prompt
            prompt = create_rag_prompt(question, context)
            
            # Generate answer
            answer = self._generate_answer(prompt)
            
            # Extract source files
            source_files = list(set([chunk.get('source', 'Unknown') for chunk in retrieved_chunks]))
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(complete_query, retrieved_chunks)
            
            logger.info("Query processed successfully")
            
            return {
                "answer": answer,
                "source_files": source_files,
                "confidence_score": confidence_score,
                "retrieved_chunks": len(retrieved_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "source_files": [],
                "confidence_score": 0.0
            }
    
    def is_model_loaded(self) -> bool:
        """Check if the LLM is loaded and ready"""
        return self.generator is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_tokens": self.max_tokens,
            "model_loaded": self.is_model_loaded()
        }
