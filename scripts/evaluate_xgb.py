# evaluate_xgb.py
from pathlib import Path
import xgboost as xgb
import pandas as pd
import pickle
import argparse
import os
import sys
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score

# Add the parent directory to the Python path so 'src' can be found
sys.path.append(str(Path(__file__).parent.parent))

import src.config as config

def evaluate_xgboost(processed_data_path, model_path):
    """Evaluates a trained XGBoost model on the test set."""
    print(" Starting XGBoost Evaluation ")
    print(f"Loading processed data from: {processed_data_path}")
    try:
        with open(processed_data_path, 'rb') as f:
            processed_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {processed_data_path}")
        return

    # Extract XGBoost specific test data
    X_test = processed_data.get('test', {}).get('xgb')
    y_test = processed_data.get('test', {}).get('y')

    if X_test is None or y_test is None:
        print("Error: XGBoost test data splits not found in processed data file.")
        return

    print(f"Test data shape: {X_test.shape}")

    print(f"Loading trained model from: {model_path}")
    try:
        model = xgb.Booster()
        model.load_model(model_path)
    except xgb.core.XGBoostError as e:
         print(f"Error loading XGBoost model from {model_path}: {e}")
         return
    except Exception as e:
         print(f"An unexpected error occurred loading the model: {e}")
         return


    # Make predictions
    print("Making predictions on test set...")
    try:
        dtest = xgb.DMatrix(X_test)
        # Use best_iteration if available, otherwise predict with full model
        best_iter = getattr(model, 'best_iteration', 0)
        if best_iter > 0:
             print(f"  Predicting using best iteration: {best_iter}")
             y_pred_proba = model.predict(dtest, iteration_range=(0, best_iter))
        else:
             print("  Predicting using full model (best iteration not found).")
             y_pred_proba = model.predict(dtest)

    except Exception as e:
         print(f"Error during prediction: {e}")
         return

    y_pred_class = (y_pred_proba > 0.5).astype(int) # Threshold at 0.5

    # Calculate metrics
    print("Calculating evaluation metrics...")
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba) # Average Precision (PR AUC)
        report = classification_report(y_test, y_pred_class, target_names=['Not Fraud', 'Fraud'])
        cm = confusion_matrix(y_test, y_pred_class)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return

    print("\n XGBoost Evaluation Results ")
    print(f"Test AUC: {auc:.4f}")
    print(f"Test Average Precision (AP): {ap:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(f"   TN: {cm[0][0]}  FP: {cm[0][1]}")
    print(f"   FN: {cm[1][0]}  TP: {cm[1][1]}")
    print("-----------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate XGBoost model.")
    parser.add_argument("--data_path", type=str, default=config.PROCESSED_DATA_PATH, help="Path to processed data pickle.")
    parser.add_argument("--model_path", type=str, default=config.XGB_MODEL_PATH, help="Path to trained XGBoost model.")
    args = parser.parse_args()

    evaluate_xgboost(args.data_path, args.model_path)
