"""
Automated launcher for RAG Chatbot
Starts FastAPI backend and Streamlit frontend automatically
"""

import subprocess
import sys
import time
import os
import threading
import signal
from pathlib import Path

def check_port(port):
    """Check if a port is already in use"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False  # Port is free
        except OSError:
            return True   # Port is in use

def run_fastapi():
    """Run FastAPI backend"""
    print("🚀 Starting FastAPI backend...")
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ FastAPI backend failed to start")
    except KeyboardInterrupt:
        print("🛑 FastAPI backend stopped")

def run_streamlit():
    """Run Streamlit frontend"""
    print("🌐 Starting Streamlit frontend...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py"
        ], check=True)
    except subprocess.CalledProcessError:
        print("❌ Streamlit frontend failed to start")
    except KeyboardInterrupt:
        print("🛑 Streamlit frontend stopped")

def wait_for_api(max_attempts=30):
    """Wait for FastAPI to be ready"""
    import requests
    
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("✅ FastAPI backend is ready!")
                return True
        except:
            pass
        
        print(f"⏳ Waiting for FastAPI... ({attempt + 1}/{max_attempts})")
        time.sleep(2)
    
    print("❌ FastAPI backend failed to start within timeout")
    return False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Shutting down services...")
    sys.exit(0)

def main():
    """Main launcher function"""
    print("🤖 RAG Chatbot Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ main.py not found. Please run from the RAG_Excel directory.")
        sys.exit(1)
    
    if not Path("streamlit_app.py").exists():
        print("❌ streamlit_app.py not found. Please run from the RAG_Excel directory.")
        sys.exit(1)
    
    # Check if ports are already in use
    if check_port(8000):
        print("⚠️  Port 8000 is already in use (FastAPI might be running)")
        choice = input("Continue anyway? (y/n): ").lower()
        if choice != 'y':
            sys.exit(1)
    
    if check_port(8501):
        print("⚠️  Port 8501 is already in use (Streamlit might be running)")
        choice = input("Continue anyway? (y/n): ").lower()
        if choice != 'y':
            sys.exit(1)
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start FastAPI in a separate thread
        fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
        fastapi_thread.start()
        
        # Wait for FastAPI to be ready
        if not wait_for_api():
            print("❌ Failed to start FastAPI backend")
            sys.exit(1)
        
        print("\n🌐 Access URLs:")
        print("📡 FastAPI Backend: http://localhost:8000")
        print("📄 API Documentation: http://localhost:8000/docs")
        print("🖥️  Streamlit Frontend: http://localhost:8501")
        print("\n💡 Press Ctrl+C to stop both services")
        print("=" * 50)
        
        # Start Streamlit (this will block until stopped)
        run_streamlit()
        
    except KeyboardInterrupt:
        print("\n🛑 Services stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
