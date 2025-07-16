#!/usr/bin/env python3
"""
Start script for Railway deployment
"""
import os
import subprocess
import sys

def main():
    port = os.environ.get('PORT', '5000')
    print(f"Starting app on port {port}")
    
    cmd = [
        'gunicorn',
        '--workers=1',
        '--timeout=300',
        f'--bind=0.0.0.0:{port}',
        'app:app'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == '__main__':
    main()
