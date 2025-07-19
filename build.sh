#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Run the optimized model extraction script
python extract_optimized_compatible.py

echo "Build completed successfully!"