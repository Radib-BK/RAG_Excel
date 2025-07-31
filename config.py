"""
Configuration management for RAG Chatbot
Centralized settings and model configurations
"""

import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Model configuration settings"""
    # Language Model Settings
    llm_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda"
    use_quantization: bool = True
    torch_dtype: str = "float16"  # float16, float32, bfloat16
    
    # Generation Settings
    max_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

@dataclass
class ProcessingConfig:
    """Document processing configuration"""
    chunk_size: int = 512
    chunk_overlap: int = 100
    top_k_results: int = 5
    max_context_length: int = 2000
    
    # OCR Settings
    ocr_language: str = "eng"
    ocr_config: str = "--psm 6"  # Page segmentation mode
    
    # File size limits (in MB)
    max_file_size_mb: int = 50
    
    # Supported file extensions
    supported_extensions: list = None
    
    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = ['.pdf', '.docx', '.txt', '.csv', '.db', '.jpg', '.jpeg', '.png']

@dataclass
class SystemConfig:
    """System configuration"""
    vector_store_path: str = "./vector_store"
    log_level: str = "INFO"
    log_file: str = "rag_chatbot.log"
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    # CORS Settings
    cors_origins: list = None
    cors_allow_credentials: bool = True
    cors_allow_methods: list = None
    cors_allow_headers: list = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]
        if self.cors_allow_methods is None:
            self.cors_allow_methods = ["*"]
        if self.cors_allow_headers is None:
            self.cors_allow_headers = ["*"]

class ConfigManager:
    """Manages configuration from environment variables and defaults"""
    
    def __init__(self):
        self.model_config = self._load_model_config()
        self.processing_config = self._load_processing_config()
        self.system_config = self._load_system_config()
    
    def _load_model_config(self) -> ModelConfig:
        """Load model configuration from environment"""
        return ModelConfig(
            llm_model_name=os.getenv('MODEL_NAME', ModelConfig.llm_model_name),
            embedding_model_name=os.getenv('EMBEDDING_MODEL', ModelConfig.embedding_model_name),
            device=os.getenv('DEVICE', ModelConfig.device),
            use_quantization=os.getenv('USE_QUANTIZATION', 'true').lower() == 'true',
            torch_dtype=os.getenv('TORCH_DTYPE', ModelConfig.torch_dtype),
            max_tokens=int(os.getenv('MAX_TOKENS', ModelConfig.max_tokens)),
            temperature=float(os.getenv('TEMPERATURE', ModelConfig.temperature)),
            top_p=float(os.getenv('TOP_P', ModelConfig.top_p)),
            do_sample=os.getenv('DO_SAMPLE', 'true').lower() == 'true'
        )
    
    def _load_processing_config(self) -> ProcessingConfig:
        """Load processing configuration from environment"""
        return ProcessingConfig(
            chunk_size=int(os.getenv('CHUNK_SIZE', ProcessingConfig.chunk_size)),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', ProcessingConfig.chunk_overlap)),
            top_k_results=int(os.getenv('TOP_K_RESULTS', ProcessingConfig.top_k_results)),
            max_context_length=int(os.getenv('MAX_CONTEXT_LENGTH', ProcessingConfig.max_context_length)),
            ocr_language=os.getenv('OCR_LANGUAGE', ProcessingConfig.ocr_language),
            ocr_config=os.getenv('OCR_CONFIG', ProcessingConfig.ocr_config),
            max_file_size_mb=int(os.getenv('MAX_FILE_SIZE_MB', ProcessingConfig.max_file_size_mb))
        )
    
    def _load_system_config(self) -> SystemConfig:
        """Load system configuration from environment"""
        return SystemConfig(
            vector_store_path=os.getenv('VECTOR_STORE_PATH', SystemConfig.vector_store_path),
            log_level=os.getenv('LOG_LEVEL', SystemConfig.log_level),
            log_file=os.getenv('LOG_FILE', SystemConfig.log_file),
            api_host=os.getenv('API_HOST', SystemConfig.api_host),
            api_port=int(os.getenv('API_PORT', SystemConfig.api_port)),
            api_reload=os.getenv('API_RELOAD', 'true').lower() == 'true'
        )
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configuration as a dictionary"""
        return {
            'model': self.model_config.__dict__,
            'processing': self.processing_config.__dict__,
            'system': self.system_config.__dict__
        }
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        try:
            # Validate model settings
            assert self.model_config.max_tokens > 0, "max_tokens must be positive"
            assert 0.0 <= self.model_config.temperature <= 2.0, "temperature must be between 0 and 2"
            assert 0.0 <= self.model_config.top_p <= 1.0, "top_p must be between 0 and 1"
            
            # Validate processing settings
            assert self.processing_config.chunk_size > 0, "chunk_size must be positive"
            assert self.processing_config.chunk_overlap >= 0, "chunk_overlap must be non-negative"
            assert self.processing_config.chunk_overlap < self.processing_config.chunk_size, "chunk_overlap must be less than chunk_size"
            assert self.processing_config.top_k_results > 0, "top_k_results must be positive"
            
            # Validate system settings
            assert self.system_config.api_port > 0, "api_port must be positive"
            
            return True
            
        except AssertionError as e:
            print(f"Configuration validation error: {e}")
            return False
    
    def print_config(self):
        """Print current configuration"""
        print("Current Configuration:")
        print("=" * 50)
        
        print("\n🤖 Model Configuration:")
        for key, value in self.model_config.__dict__.items():
            print(f"  {key}: {value}")
        
        print("\n📄 Processing Configuration:")
        for key, value in self.processing_config.__dict__.items():
            print(f"  {key}: {value}")
        
        print("\n⚙️ System Configuration:")
        for key, value in self.system_config.__dict__.items():
            print(f"  {key}: {value}")

# Global configuration instance
config = ConfigManager()

# Alternative model configurations for easy switching
ALTERNATIVE_MODELS = {
    "tiny": {
        "llm_model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2"
    },
    "small": {
        "llm_model_name": "microsoft/DialoGPT-small",
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2"
    },
    "medium": {
        "llm_model_name": "microsoft/phi-2",
        "embedding_model_name": "sentence-transformers/all-mpnet-base-v2"
    },
    "cpu_optimized": {
        "llm_model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu",
        "use_quantization": False,
        "torch_dtype": "float32"
    }
}

def apply_model_preset(preset_name: str):
    """Apply a predefined model configuration preset"""
    if preset_name not in ALTERNATIVE_MODELS:
        available = ", ".join(ALTERNATIVE_MODELS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    preset = ALTERNATIVE_MODELS[preset_name]
    
    # Update environment variables
    for key, value in preset.items():
        env_key = key.upper()
        if env_key == "LLM_MODEL_NAME":
            env_key = "MODEL_NAME"
        elif env_key == "EMBEDDING_MODEL_NAME":
            env_key = "EMBEDDING_MODEL"
        
        os.environ[env_key] = str(value)
    
    # Reload configuration
    global config
    config = ConfigManager()
    print(f"Applied model preset: {preset_name}")
    config.print_config()
