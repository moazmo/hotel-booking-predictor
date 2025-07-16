#!/usr/bin/env python3
"""
Setup and run script for Hotel Booking Predictor
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dataset():
    """Check if the dataset file exists."""
    dataset_file = "../first intern project.csv"
    if os.path.exists(dataset_file):
        print("✅ Dataset file found")
        return True
    else:
        print("❌ Dataset file 'first intern project.csv' not found in parent directory")
        print("Please ensure the dataset is available before running the model extraction")
        return False

def create_virtual_environment():
    """Create a virtual environment."""
    venv_path = "venv"
    if os.path.exists(venv_path):
        print("✅ Virtual environment already exists")
        return True
    
    try:
        print("🔧 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
        print("✅ Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def install_dependencies():
    """Install required dependencies."""
    try:
        print("📦 Installing dependencies...")
        
        # Determine the correct pip path based on OS
        if os.name == 'nt':  # Windows
            pip_path = os.path.join("venv", "Scripts", "pip")
        else:  # Unix-like systems
            pip_path = os.path.join("venv", "bin", "pip")
        
        # Install requirements
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("You may need to install them manually:")
        print("pip install -r requirements.txt")
        return False

def extract_model():
    """Extract and train the model."""
    if not check_dataset():
        print("⚠️ Skipping model extraction due to missing dataset")
        return False
    
    try:
        print("🤖 Extracting and training model...")
        print("This may take a few minutes...")
        
        # Determine the correct python path
        if os.name == 'nt':  # Windows
            python_path = os.path.join("venv", "Scripts", "python")
        else:  # Unix-like systems
            python_path = os.path.join("venv", "bin", "python")
        
        subprocess.run([python_path, "extract_model.py"], check=True)
        print("✅ Model extraction completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Model extraction failed: {e}")
        return False

def run_application():
    """Run the Flask application."""
    try:
        print("🚀 Starting the application...")
        print("Access the app at: http://localhost:5000")
        print("Press Ctrl+C to stop the application")
        
        # Determine the correct python path
        if os.name == 'nt':  # Windows
            python_path = os.path.join("venv", "Scripts", "python")
        else:  # Unix-like systems
            python_path = os.path.join("venv", "bin", "python")
        
        subprocess.run([python_path, "app.py"])
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start application: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")

def main():
    """Main setup and run function."""
    print("🏨 Hotel Booking Predictor Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create virtual environment
    if not create_virtual_environment():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    # Extract model
    model_extracted = extract_model()
    
    if not model_extracted:
        print("\n⚠️ Model extraction failed or skipped.")
        print("You can still run the app, but predictions won't work until you:")
        print("1. Ensure 'first intern project.csv' is in the parent directory")
        print("2. Run: python extract_model.py")
        
        response = input("\nDo you want to start the app anyway? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    # Run the application
    print("\n" + "=" * 40)
    run_application()
    
    return 0

if __name__ == "__main__":
    exit(main())
