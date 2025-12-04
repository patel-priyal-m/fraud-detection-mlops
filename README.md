# Fraud Detection MLOps Project

A real-time fraud detection system demonstrating MLOps principles with model training, API serving, live monitoring, and containerization.

## Tech Stack

- **Python 3.9+**
- **MLflow** - Experiment tracking and model registry
- **FastAPI** - REST API for model serving
- **Streamlit** - Real-time monitoring dashboard
- **Docker** - Containerization
- **Scikit-learn** - Machine learning (RandomForest)

## Project Structure

```
fraud-detection-mlops/
├── train.py           # Model training with MLflow tracking
├── main.py            # FastAPI prediction server
├── dashboard.py       # Streamlit monitoring dashboard
├── simulate.py        # Traffic generator for testing
├── Dockerfile         # Container configuration
├── requirements.txt   # Python dependencies
├── data/              # Dataset folder (add creditcard.csv here)
└── logs/              # API prediction logs
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/patel-priyal-m/fraud-detection-mlops.git
cd fraud-detection-mlops
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` folder.

### 5. Train the model

```bash
python train.py
```

This will:
- Load and preprocess the data
- Train a RandomForest model
- Log parameters and metrics to MLflow
- Register the model in MLflow registry

## Running the Application

You need 3 terminals to run the full demo:

### Terminal 1: Start the API Server

```bash
python main.py
```

API will be available at: http://localhost:8000

- Swagger docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Terminal 2: Start the Dashboard

```bash
streamlit run dashboard.py
```

Dashboard will be available at: http://localhost:8501

### Terminal 3: Run the Traffic Simulator

```bash
python simulate.py
```

This sends 20 sample transactions to the API with 1-second delays.

## API Usage

### Predict Endpoint

```bash
POST /predict
Content-Type: application/json

{
    "features": [0.0, 0.0, ..., 0.0]  # 29 float values (V1-V28 + Amount_Scaled)
}
```

Response:

```json
{
    "fraud": false,
    "probability": 0.02,
    "timestamp": "2025-12-04T16:59:30"
}
```

## Docker

### Build the image

```bash
docker build -t fraud-detection-api .
```

### Run the container

```bash
docker run -p 8000:8000 fraud-detection-api
```

## MLflow UI

View experiment tracking:

```bash
mlflow ui
```

Open: http://localhost:5000

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 99.93% |
| Precision | 78.64% |
| Recall | 82.65% |
| F1 Score | 80.60% |

