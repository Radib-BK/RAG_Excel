"""
Installation and setup script for RAG Chatbot
Automates the installation process and dependency checks
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("   Please install Python 3.8 or higher")
        return False

def check_venv():
    """Check if virtual environment is available"""
    try:
        # Check if we're in a virtual environment
        import sys
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("✅ Virtual environment is active")
            return True
        else:
            print("⚠️ No virtual environment detected")
            print("   Recommendation: Create a virtual environment")
            print("   python -m venv ragbot-env")
            print("   ragbot-env\\Scripts\\activate  # Windows")
            print("   source ragbot-env/bin/activate  # Linux/macOS")
            return False
    except Exception:
        print("ℹ️ Could not detect virtual environment status")
        return False

def install_tesseract():
    """Install or check Tesseract OCR"""
    system = platform.system()
    
    try:
        subprocess.run(["tesseract", "--version"], check=True, capture_output=True)
        print("✅ Tesseract OCR is already installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ Tesseract OCR not found")
        
        if system == "Windows":
            print("   Please download and install Tesseract from:")
            print("   https://github.com/UB-Mannheim/tesseract/wiki")
            print("   Add it to your PATH: C:\\Program Files\\Tesseract-OCR")
        elif system == "Darwin":  # macOS
            print("   Install with: brew install tesseract")
        elif system == "Linux":
            print("   Install with: sudo apt-get install tesseract-ocr")
        
        return False

def create_virtual_environment():
    """Create Python virtual environment"""
    env_name = "ragbot-env"
    
    print(f"🐍 Creating Python virtual environment: {env_name}")
    
    # Check if environment already exists
    if Path(env_name).exists():
        print(f"✅ Virtual environment '{env_name}' already exists")
        return True
    
    # Create environment
    return run_command(
        f"python -m venv {env_name}",
        f"Creating virtual environment '{env_name}'"
    )

def install_pytorch():
    """Install PyTorch with CUDA support"""
    print("🔥 Installing PyTorch with CUDA support...")
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ PyTorch with CUDA already installed")
            return True
    except ImportError:
        pass
    
    # Install PyTorch
    command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    return run_command(command, "Installing PyTorch with CUDA")

def install_requirements():
    """Install Python requirements"""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    return run_command(
        "pip install -r requirements.txt",
        "Installing Python requirements"
    )

def create_directories():
    """Create necessary directories"""
    directories = ["vector_store", "sample_files"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    return True

def setup_environment_file():
    """Setup .env file if it doesn't exist"""
    env_file = Path(".env")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    env_content = """# RAG Chatbot Configuration
MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=100
TOP_K_RESULTS=5
MAX_TOKENS=150
DEVICE=cuda
VECTOR_STORE_PATH=./vector_store
"""
    
    with open(env_file, "w") as f:
        f.write(env_content)
    
    print("📝 Created .env file with default settings")
    return True

def download_models():
    """Pre-download models to avoid first-run delays"""
    print("📥 Pre-downloading models (this may take a while)...")
    
    try:
        # Download embedding model
        print("   Downloading embedding model...")
        from sentence_transformers import SentenceTransformer
        SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("   ✅ Embedding model downloaded")
        
        # Download LLM
        print("   Downloading language model...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
        AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
        print("   ✅ Language model downloaded")
        
        return True
        
    except Exception as e:
        print(f"   ⚠️ Model download failed: {e}")
        print("   Models will be downloaded on first use")
        return False

def run_tests():
    """Run basic tests"""
    if not Path("test_api.py").exists():
        print("⚠️ test_api.py not found - skipping tests")
        return True
    
    print("🧪 Running basic tests...")
    print("   Note: This requires the server to be running")
    print("   Start the server first with: python main.py")
    
    return True

def main():
    """Main installation process"""
    print("🚀 RAG Chatbot Installation Script")
    print("=" * 50)
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Checking virtual environment", check_venv),
        ("Checking Tesseract OCR", install_tesseract),
        ("Installing PyTorch", install_pytorch),
        ("Installing requirements", install_requirements),
        ("Creating directories", create_directories),
        ("Setting up environment file", setup_environment_file),
        ("Pre-downloading models", download_models),
    ]
    
    failed_steps = []
    
    for description, function in steps:
        print(f"\n{description}...")
        try:
            if not function():
                failed_steps.append(description)
        except Exception as e:
            print(f"❌ {description} failed with error: {e}")
            failed_steps.append(description)
    
    print("\n" + "=" * 50)
    print("🏁 Installation Summary")
    print("=" * 50)
    
    if not failed_steps:
        print("✅ All steps completed successfully!")
        print("\n🎉 Installation complete!")
        print("\nNext steps:")
        print("1. Start the server: python main.py")
        print("2. Open browser: http://localhost:8000")
        print("3. Run tests: python test_api.py")
        print("4. Start Streamlit: streamlit run streamlit_app.py")
    else:
        print(f"⚠️ {len(failed_steps)} steps failed:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\nPlease address the failed steps manually.")
    
    print("\n📚 Documentation: Check README.md for detailed instructions")

if __name__ == "__main__":
    main()
