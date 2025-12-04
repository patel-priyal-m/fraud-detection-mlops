"""
train.py - Model Training Script with MLflow Tracking

This script demonstrates MLOps principles:
1. Data loading and preprocessing
2. Model training with hyperparameter tuning
3. Experiment tracking with MLflow
4. Model versioning and artifact logging

MLflow UI: After training, run `mlflow ui` to visualize experiments
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the credit card fraud dataset.
    
    The dataset contains:
    - Time: Seconds elapsed between each transaction and first transaction
    - V1-V28: PCA transformed features (anonymized for privacy)
    - Amount: Transaction amount
    - Class: 0 = Normal, 1 = Fraud
    """
    print(f"📂 Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df):,} transactions")
    print(f"   - Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.2f}%)")
    print(f"   - Normal cases: {(df['Class']==0).sum():,}")
    return df


def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the data for training.
    
    Why we do this:
    1. Scale 'Amount' - it varies wildly (0 to 25,000+)
    2. Drop 'Time' - not useful for fraud detection patterns
    3. Stratified split - maintains fraud ratio in train/test sets
       (Critical for imbalanced datasets!)
    """
    print("\n🔧 Preprocessing data...")
    
    # Scale the Amount feature
    scaler = StandardScaler()
    df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
    
    # Features and target
    # Drop Time (not predictive) and original Amount (we use scaled)
    feature_cols = [col for col in df.columns if col not in ['Class', 'Time', 'Amount']]
    X = df[feature_cols]
    y = df['Class']
    
    print(f"   - Features: {len(feature_cols)} columns")
    print(f"   - Target: 'Class' (0=Normal, 1=Fraud)")
    
    # Stratified split to maintain class distribution
    # This is CRITICAL for imbalanced datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y  # Ensures same fraud ratio in train and test
    )
    
    print(f"   - Train set: {len(X_train):,} samples")
    print(f"   - Test set: {len(X_test):,} samples")
    
    return X_train, X_test, y_train, y_test, feature_cols


def train_model(X_train, y_train, n_estimators: int = 100):
    """
    Train a Random Forest Classifier.
    
    Why Random Forest?
    - Handles imbalanced data reasonably well
    - No feature scaling required (tree-based)
    - Provides feature importance
    - Less prone to overfitting than single decision tree
    
    class_weight='balanced' automatically adjusts weights
    inversely proportional to class frequencies.
    """
    print(f"\n🤖 Training Random Forest with {n_estimators} estimators...")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight='balanced',  # Handle imbalanced classes
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
        max_depth=10,  # Prevent overfitting
        min_samples_split=10
    )
    
    model.fit(X_train, y_train)
    print("✅ Model training complete!")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance with multiple metrics.
    
    For fraud detection, we care about:
    - Precision: Of predicted frauds, how many are actual frauds?
    - Recall: Of actual frauds, how many did we catch?
    - F1 Score: Balance between precision and recall
    
    High recall is crucial - we don't want to miss frauds!
    """
    print("\n📊 Evaluating model...")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of fraud
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    print(f"   - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   - Precision: {metrics['precision']:.4f}")
    print(f"   - Recall:    {metrics['recall']:.4f}")
    print(f"   - F1 Score:  {metrics['f1_score']:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   Confusion Matrix:")
    print(f"   TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"   FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
    
    return metrics


def main():
    """
    Main training pipeline with MLflow tracking.
    
    MLflow tracks:
    - Parameters: Hyperparameters used for training
    - Metrics: Model performance metrics
    - Artifacts: The trained model file
    - Model: Registered model for deployment
    """
    # Configuration
    DATA_PATH = "data/creditcard.csv"
    N_ESTIMATORS = 100
    EXPERIMENT_NAME = "fraud-detection"
    
    # Set MLflow experiment
    # This creates a folder 'mlruns' to store all experiments
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    print("=" * 60)
    print("🚀 FRAUD DETECTION MODEL TRAINING")
    print("=" * 60)
    
    # Load and preprocess data
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, feature_cols = preprocess_data(df)
    
    # Start MLflow run
    # Everything inside this block is tracked
    with mlflow.start_run(run_name="random-forest-baseline"):
        
        print("\n📝 MLflow tracking started...")
        
        # Log parameters (inputs to the model)
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("n_features", len(feature_cols))
        
        # Train model
        model = train_model(X_train, y_train, N_ESTIMATORS)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log metrics (outputs/results)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log the model artifact
        # This saves the model in MLflow format for easy loading later
        print("\n💾 Saving model to MLflow...")
        mlflow.sklearn.log_model(
            model, 
            artifact_path="model",
            registered_model_name="fraud-detection-model"
        )
        
        # Get the run ID for reference
        run_id = mlflow.active_run().info.run_id
        print(f"\n✅ MLflow Run ID: {run_id}")
        print(f"   Model saved at: mlruns/")
        
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print("\n📌 Next steps:")
    print("   1. Run 'mlflow ui' to view experiment dashboard")
    print("   2. Run 'python main.py' to start the API server")
    print("   3. The model is ready for serving!")


if __name__ == "__main__":
    main()
