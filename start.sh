#!/bin/bash
PORT=${PORT:-5000}
echo "Starting app on port $PORT"
exec gunicorn --workers=1 --timeout=300 --bind=0.0.0.0:$PORT app:app
