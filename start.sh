#!/bin/bash

# Railway startup script for Hotel Booking Predictor
echo "🚀 Starting Hotel Booking Predictor on Railway..."

# Set default PORT if not provided
PORT=${PORT:-5000}
echo "🔍 PORT environment variable: $PORT"

# Ensure models are created
echo "📊 Creating ML models..."
python extract_model.py

# Start the application with gunicorn using Railway's recommended pattern
echo "🌐 Starting gunicorn server on port $PORT..."
exec gunicorn --bind 0.0.0.0:$PORT app:app
