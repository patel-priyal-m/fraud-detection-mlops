"""
main.py - FastAPI Server for Fraud Detection

This API serves the trained MLflow model and provides:
- POST /predict: Accept transaction features, return fraud prediction
- GET /health: Health check endpoint

The API logs all predictions to logs/api_logs.csv for monitoring.
"""

import os
import csv
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "fraud-detection-model"
MODEL_VERSION = "1"
LOGS_DIR = "logs"
LOGS_FILE = os.path.join(LOGS_DIR, "api_logs.csv")

# Feature names expected by the model (V1-V28 + Amount_Scaled)
FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Amount_Scaled"]


# ---------------------------------------------------------------------------
# Pydantic Models (Request/Response Schemas)
# ---------------------------------------------------------------------------
class Transaction(BaseModel):
    """
    Input schema for a single transaction.
    
    Fields:
    - features: List of 29 float values (V1-V28 + Amount_Scaled)
    
    Note: In production, you'd have named fields for each feature.
    We use a list here for simplicity with the creditcard.csv format.
    """
    features: List[float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.0] * 29  # 29 features
            }
        }


class PredictionResponse(BaseModel):
    """
    Output schema for prediction response.
    
    Fields:
    - fraud: Boolean indicating if transaction is predicted as fraud
    - probability: Float between 0-1 indicating fraud likelihood
    - timestamp: When the prediction was made
    """
    fraud: bool
    probability: float
    timestamp: str


# ---------------------------------------------------------------------------
# Global Model Variable
# ---------------------------------------------------------------------------
model = None


# ---------------------------------------------------------------------------
# Startup/Shutdown Events
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    
    On startup:
    - Load the trained model from MLflow registry
    - Create logs directory if it doesn't exist
    - Initialize the CSV log file with headers
    
    On shutdown:
    - Cleanup resources if needed
    """
    global model
    
    print("[INFO] Starting up FastAPI server...")
    
    # Create logs directory if it doesn't exist
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Initialize CSV log file with headers if it doesn't exist
    if not os.path.exists(LOGS_FILE):
        with open(LOGS_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'prediction', 'probability'])
        print(f"[INFO] Created log file: {LOGS_FILE}")
    
    # Load model from MLflow Model Registry
    print(f"[INFO] Loading model: {MODEL_NAME} (version {MODEL_VERSION})...")
    
    try:
        # Load from MLflow Model Registry
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"[INFO] Model loaded successfully from: {model_uri}")
    except Exception as e:
        print(f"[ERROR] Failed to load model from registry: {e}")
        print("[INFO] Attempting to load from latest run...")
        
        # Fallback: Load from the latest run in the experiment
        try:
            experiment = mlflow.get_experiment_by_name("fraud-detection")
            if experiment:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=1
                )
                if not runs.empty:
                    run_id = runs.iloc[0]['run_id']
                    model_uri = f"runs:/{run_id}/model"
                    model = mlflow.sklearn.load_model(model_uri)
                    print(f"[INFO] Model loaded from run: {run_id}")
        except Exception as e2:
            print(f"[ERROR] Failed to load model: {e2}")
            raise RuntimeError("Could not load model. Please run train.py first.")
    
    yield  # Server is running
    
    # Shutdown
    print("[INFO] Shutting down FastAPI server...")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection using ML model served via MLflow",
    version="1.0.0",
    lifespan=lifespan
)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def log_prediction(timestamp: str, prediction: int, probability: float):
    """
    Append prediction to CSV log file.
    
    This mimics a database insert for the dashboard to read.
    In production, you'd use a real database (PostgreSQL, MongoDB, etc.)
    """
    with open(LOGS_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, prediction, probability])


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Fraud Detection API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of the API and model.
    Useful for container orchestration (Kubernetes, Docker Swarm).
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    """
    Predict if a transaction is fraudulent.
    
    Accepts:
    - transaction: JSON with 'features' list (29 float values)
    
    Returns:
    - fraud: True if predicted as fraud
    - probability: Confidence score (0.0 to 1.0)
    - timestamp: When prediction was made
    
    Example request:
    ```json
    {
        "features": [-1.3598, -0.0728, 2.5363, ..., 0.0, 149.62]
    }
    ```
    """
    global model
    
    # Validate model is loaded
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please ensure train.py has been run."
        )
    
    # Validate feature count
    if len(transaction.features) != 29:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 29 features, got {len(transaction.features)}. "
                   f"Features should be: V1-V28 + Amount_Scaled"
        )
    
    try:
        # Prepare input data as DataFrame (model expects this format)
        input_data = pd.DataFrame([transaction.features], columns=FEATURE_NAMES)
        
        # Get prediction and probability
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]  # Probability of class 1 (fraud)
        
        # Create timestamp
        timestamp = datetime.now().isoformat()
        
        # Log to CSV for dashboard
        log_prediction(timestamp, int(prediction), float(probability))
        
        # Return response
        return PredictionResponse(
            fraud=bool(prediction),
            probability=float(probability),
            timestamp=timestamp
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ---------------------------------------------------------------------------
# Run with Uvicorn (for development)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
