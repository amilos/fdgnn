# train_xgb.py
import xgboost as xgb
import pandas as pd
import pickle
import joblib
import argparse
import os
import sys
from pathlib import Path
from sklearn.metrics import roc_auc_score

# Add the parent directory to the Python path so 'src' can be found
sys.path.append(str(Path(__file__).parent.parent))

import src.config as config


def train_xgboost(processed_data_path, model_output_path):
    """Trains an XGBoost model."""
    print("--- Starting XGBoost Training ---")
    print(f"Loading processed data from: {processed_data_path}")
    try:
        with open(processed_data_path, 'rb') as f:
            processed_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {processed_data_path}")
        print("Please run preprocess_main.py first.")
        return

    # Extract XGBoost specific data splits
    X_train = processed_data.get('train', {}).get('xgb')
    y_train = processed_data.get('train', {}).get('y')
    X_val = processed_data.get('val', {}).get('xgb')
    y_val = processed_data.get('val', {}).get('y')

    if X_train is None or y_train is None or X_val is None or y_val is None:
        print("Error: XGBoost data splits not found in processed data file.")
        return

    print(f"Train data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    # Convert to DMatrix
    print("Converting data to DMatrix...")
    # Enable categorical features if using native support and data prepared accordingly
    # enable_cat = 'encoder_xgb' in joblib.load(config.PROCESSORS_PATH) # Example check
    dtrain = xgb.DMatrix(X_train, label=y_train) #, enable_categorical=enable_cat)
    dval = xgb.DMatrix(X_val, label=y_val) #, enable_categorical=enable_cat)

    watchlist = [(dtrain, 'train'), (dval, 'eval')]

    print("Training XGBoost model...")
    params = config.XGB_PARAMS
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=config.XGB_NUM_BOOST_ROUND,
        evals=watchlist,
        early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS,
        verbose_eval=50 # Print progress every 50 rounds
    )

    print(f"Best AUC: {model.best_score:.4f} at iteration {model.best_iteration}")

    # Ensure model directory exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    print(f"Saving trained model to: {model_output_path}")
    model.save_model(model_output_path) # Use native save_model

    print("--- XGBoost Training Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model.")
    parser.add_argument("--data_path", type=str, default=config.PROCESSED_DATA_PATH, help="Path to processed data pickle.")
    parser.add_argument("--model_path", type=str, default=config.XGB_MODEL_PATH, help="Output path for trained XGBoost model.")
    args = parser.parse_args()

    train_xgboost(args.data_path, args.model_path)