#!/bin/bash
echo "Building Hotel Booking Predictor..."
echo "Installing Python dependencies..."
pip install -r requirements.txt
echo "Creating ML models..."
python extract_model.py
echo "Build complete!"
