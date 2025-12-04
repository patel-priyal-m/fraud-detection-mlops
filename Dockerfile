# Dockerfile for Fraud Detection API
# 
# This containerizes the FastAPI server for deployment.
# 
# Build: docker build -t fraud-detection-api .
# Run:   docker run -p 8000:8000 fraud-detection-api

# ---------------------------------------------------------------------------
# Base Image
# ---------------------------------------------------------------------------
# Using slim Python image to reduce size
FROM python:3.11-slim

# ---------------------------------------------------------------------------
# Set Working Directory
# ---------------------------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------------------------
# Install Dependencies
# ---------------------------------------------------------------------------
# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Copy Application Files
# ---------------------------------------------------------------------------
# Copy the application code
COPY main.py .
COPY train.py .

# Copy MLflow model artifacts (needed for serving)
COPY mlruns/ ./mlruns/

# Create logs directory
RUN mkdir -p logs

# ---------------------------------------------------------------------------
# Expose Port
# ---------------------------------------------------------------------------
EXPOSE 8000

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# ---------------------------------------------------------------------------
# Run the Application
# ---------------------------------------------------------------------------
# Using uvicorn to serve the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
