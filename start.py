#!/usr/bin/env python3
"""
Start script for Railway deployment
"""
import os
import sys

def main():
    # Get port from environment, default to 5000
    port = os.environ.get('PORT', '5000')
    
    # Validate port is numeric
    try:
        port_num = int(port)
        if port_num < 1 or port_num > 65535:
            port = '5000'
    except (ValueError, TypeError):
        port = '5000'
    
    print(f"Starting app on port {port}")
    
    # Import and run the Flask app directly
    from app import app
    app.run(host='0.0.0.0', port=int(port), debug=False)

if __name__ == '__main__':
    main()
