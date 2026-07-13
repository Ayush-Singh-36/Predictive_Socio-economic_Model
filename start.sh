#!/bin/bash

# Start the FastAPI backend in the background (&)
echo "Starting FastAPI backend serving engine..."
uvicorn app:app --host 0.0.0.0 --port 8000 &

# Wait briefly for the backend process to fully bind to port 8000
sleep 3

# Start the Streamlit frontend in the foreground
echo "Starting Streamlit web dashboard..."
streamlit run frontend.py --server.port=8501 --server.address=0.0.0.0