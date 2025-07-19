#!/bin/bash
# Simple build script for Render deployment
echo "Starting build process..."

# Install Python dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if the extract script exists and run it
if [ -f "extract_optimized_compatible.py" ]; then
    echo "Running optimized model extraction..."
    python extract_optimized_compatible.py
elif [ -f "extract_model.py" ]; then
    echo "Running basic model extraction..."
    python extract_model.py
else
    echo "⚠️  No model extraction script found. App will create models on first run."
fi

echo "✅ Build completed successfully!"
