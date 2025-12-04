"""
simulate.py - Traffic Generator for Fraud Detection API

This script simulates real-time transactions by:
1. Loading sample data from creditcard.csv
2. Sending POST requests to the API endpoint
3. Adding delays between requests to simulate real traffic

Usage: python simulate.py
"""

import time
import requests
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = "http://localhost:8000/predict"
DATA_PATH = "data/creditcard.csv"
NUM_SAMPLES = 20
DELAY_SECONDS = 1

# Feature columns (V1-V28 + Amount, which we'll scale)
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def load_sample_data(filepath: str, num_samples: int) -> pd.DataFrame:
    """
    Load a random sample of transactions from the dataset.
    
    We intentionally sample some fraud cases to make the demo interesting.
    """
    print(f"[INFO] Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Get some fraud and normal transactions for a good mix
    fraud_df = df[df['Class'] == 1].sample(min(5, len(df[df['Class'] == 1])), random_state=42)
    normal_df = df[df['Class'] == 0].sample(num_samples - len(fraud_df), random_state=42)
    
    # Combine and shuffle
    sample_df = pd.concat([fraud_df, normal_df]).sample(frac=1, random_state=42)
    
    print(f"[INFO] Loaded {len(sample_df)} samples ({len(fraud_df)} fraud, {len(normal_df)} normal)")
    return sample_df


def prepare_features(row: pd.Series) -> list:
    """
    Prepare features for API request.
    
    The API expects 29 features: V1-V28 + Amount_Scaled
    We do a simple scaling of Amount here (divide by max typical amount)
    """
    features = []
    
    # Add V1-V28
    for i in range(1, 29):
        features.append(float(row[f"V{i}"]))
    
    # Add scaled Amount (simple normalization)
    # In production, you'd use the same scaler from training
    amount_scaled = (row["Amount"] - 88.35) / 250.12  # Approximate mean/std from training
    features.append(float(amount_scaled))
    
    return features


def send_transaction(features: list) -> dict:
    """
    Send a transaction to the API and return the response.
    """
    payload = {"features": features}
    
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
        return None


def main():
    """
    Main simulation loop.
    """
    print("=" * 60)
    print("FRAUD DETECTION - TRAFFIC SIMULATOR")
    print("=" * 60)
    print(f"[INFO] Target API: {API_URL}")
    print(f"[INFO] Sending {NUM_SAMPLES} transactions with {DELAY_SECONDS}s delay")
    print("=" * 60)
    
    # Load sample data
    df = load_sample_data(DATA_PATH, NUM_SAMPLES)
    
    # Check if API is running
    try:
        health = requests.get("http://localhost:8000/health", timeout=5)
        if health.status_code != 200:
            print("[ERROR] API health check failed. Is the server running?")
            return
        print("[INFO] API is healthy. Starting simulation...\n")
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to API. Please start the server first:")
        print("        python main.py")
        return
    
    # Send transactions
    fraud_detected = 0
    total_sent = 0
    
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        # Prepare features
        features = prepare_features(row)
        actual_fraud = int(row['Class'])
        
        # Send request
        print(f"[{idx}/{NUM_SAMPLES}] Sending transaction (Actual: {'FRAUD' if actual_fraud else 'Normal'})...", end=" ")
        
        result = send_transaction(features)
        
        if result:
            predicted_fraud = result['fraud']
            probability = result['probability']
            
            status = "FRAUD DETECTED!" if predicted_fraud else "Normal"
            print(f"Prediction: {status} (Prob: {probability*100:.2f}%)")
            
            if predicted_fraud:
                fraud_detected += 1
            total_sent += 1
        else:
            print("Failed!")
        
        # Delay before next request
        if idx < len(df):
            time.sleep(DELAY_SECONDS)
    
    # Summary
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"[INFO] Total transactions sent: {total_sent}")
    print(f"[INFO] Fraud predictions: {fraud_detected}")
    print(f"[INFO] Check the dashboard at: http://localhost:8501")


if __name__ == "__main__":
    main()
