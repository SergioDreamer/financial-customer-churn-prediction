#!/bin/bash

# Install required packages
pip install fastapi uvicorn

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the FastAPI application
python src/api/main.py
