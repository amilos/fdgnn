# infer_xgb.py
from pathlib import Path
import xgboost as xgb
import pandas as pd
import joblib
import argparse
import os
import numpy as np
import sys

# Add the parent directory to the Python path so 'src' can be found
sys.path.append(str(Path(__file__).parent.parent))

import src.config as config

# Import necessary functions from preprocess_utils
from preprocess_utils import preprocess_for_inference, load



def infer_xgboost(raw_trans_path, raw_id_path, processors_path, model_path, output_path):
    """Runs inference using a trained XGBoost model on new raw data."""
    print("--- Starting XGBoost Inference ---")

    # 1. Load new raw data
    print(f"Loading new raw data from: {raw_trans_path}, {raw_id_path}")
    try:
        df_new_raw = load(raw_trans_path, raw_id_path)
        if config.ID_COL not in df_new_raw.columns:
             print(f"Warning: ID column '{config.ID_COL}' not found. Using index.")
             original_ids = pd.Series(range(len(df_new_raw)), name="index")
        else:
             original_ids = df_new_raw[config.ID_COL].copy()
    except FileNotFoundError:
        print("Error: Raw inference file(s) not found.")
        return
    except Exception as e:
        print(f"Error loading raw inference data: {e}")
        return

    # 2. Preprocess new data using saved processors
    # This function applies transformations but might return extra columns
    print("Preprocessing new data...")
    try:
        processors = joblib.load(processors_path)
        num_cols = processors.get('final_num_cols')
        cat_cols = processors.get('final_cat_cols')
        if num_cols is None or cat_cols is None:
             print("Warning: Final column lists not found in processors. Using defaults.")
             num_cols = config.DEFAULT_NUM_COLUMNS + config.VESTA_COLUMNS
             cat_cols = config.DEFAULT_CAT_COLUMNS
        exclude_cols = config.DEFAULT_EXCLUDED_COLUMNS

        df_new_processed = preprocess_for_inference(
            df_new=df_new_raw,
            processor_path=processors_path,
            model_type='xgboost',
            num_cols=num_cols,
            cat_cols=cat_cols,
            exclude_cols_initial=exclude_cols # Pass exclude_cols if preprocess_for_inference uses it
        )
    except FileNotFoundError:
         print(f"Error: Processors file not found at {processors_path}")
         return
    except Exception as e:
         print(f"Error during inference preprocessing: {e}")
         # It's possible the error originates in preprocess_for_inference or its sub-functions
         # Ensure encode_categoricals(fit=False) correctly handles all cat_cols
         return

    # 3. Load trained model BEFORE creating DMatrix
    print(f"Loading trained model from: {model_path}")
    try:
        model = xgb.Booster()
        model.load_model(model_path)
        # --- Get feature names the model expects ---
        # Note: feature_names might be None if model saved in older format or without them
        expected_features = model.feature_names
        if expected_features is None:
            # Fallback: Try loading from processors if saved there, or raise error
            # encoder_xgb = processors.get('encoder_xgb')
            # if encoder_xgb and 'output_features' in encoder_xgb: # Assuming you stored them
            #      expected_features = encoder_xgb['output_features']
            # else:
            raise ValueError("Cannot determine expected feature names from the loaded XGBoost model. "
                             "Ensure the model was saved with feature names, or store them in processors.")
        print(f"   Model expects {len(expected_features)} features.")

    except xgb.core.XGBoostError as e:
         print(f"Error loading XGBoost model from {model_path}: {e}")
         return
    except Exception as e:
         print(f"An unexpected error occurred loading the model or getting features: {e}")
         return

    # 4. Select ONLY the features the model expects from the processed data
    print("Selecting final features for prediction...")
    try:
        # Check if all expected features are present in the processed dataframe
        missing_model_features = set(expected_features) - set(df_new_processed.columns)
        if missing_model_features:
            raise ValueError(f"Processed data is missing columns the model expects: {missing_model_features}")

        df_final_features = df_new_processed[expected_features]

        # --- Verification Step (Optional but Recommended) ---
        print("   Verifying final feature dtypes...")
        invalid_dtypes = df_final_features.select_dtypes(exclude=[np.number, 'bool']).columns # Allow bool too
        if not invalid_dtypes.empty:
             print(f"Error: Final selected features still contain non-numerical dtypes: {invalid_dtypes.tolist()}")
             print("This indicates an issue in the 'encode_categoricals(fit=False)' step within preprocess_for_inference.")
             print("Ensure all categorical features were correctly transformed to numerical representations.")
             # Example: Print problematic columns and their dtypes
             # print(df_final_features[invalid_dtypes].dtypes)
             return # Stop before creating DMatrix
        print("   Dtypes verified.")
        # --- End Verification ---

    except KeyError as e:
        print(f"Error: Column mismatch. Model expects feature '{e}' which is not in the preprocessed data.")
        print(f"   Available columns: {df_new_processed.columns.tolist()}")
        return
    except Exception as e:
        print(f"Error selecting final features: {e}")
        return


    # 5. Make predictions using the *filtered* DataFrame
    print("Making predictions...")
    try:
        # Create DMatrix ONLY from the selected, verified features
        dnew = xgb.DMatrix(df_final_features)
        best_iter = getattr(model, 'best_iteration', 0)
        if best_iter > 0:
             predictions_proba = model.predict(dnew, iteration_range=(0, best_iter))
        else:
             predictions_proba = model.predict(dnew)
    except Exception as e:
         print(f"Error during prediction: {e}")
         return


    # 6. Format and save output
    output_df = pd.DataFrame({
        config.ID_COL: original_ids,
        config.TARGET_COL: predictions_proba
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving predictions to: {output_path}")
    try:
        output_df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"Error saving predictions: {e}")

    print("--- XGBoost Inference Finished ---")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XGBoost inference.")
    parser.add_argument("--raw_trans", type=str, default=config.TRANSACTION_FILE_TEST, help="Path to new raw transaction data.")
    parser.add_argument("--raw_id", type=str, default=config.IDENTITY_FILE_TEST, help="Path to new raw identity data.")
    parser.add_argument("--proc_path", type=str, default=config.PROCESSORS_PATH, help="Path to saved processors joblib.")
    parser.add_argument("--model_path", type=str, default=config.XGB_MODEL_PATH, help="Path to trained XGBoost model.")
    parser.add_argument("--out_path", type=str, default=os.path.join(config.PREDICTION_DIR, "xgb_predictions.csv"), help="Output path for predictions CSV.")
    args = parser.parse_args()

    # Create prediction directory if it doesn't exist
    os.makedirs(config.PREDICTION_DIR, exist_ok=True)

    infer_xgboost(args.raw_trans, args.raw_id, args.proc_path, args.model_path, args.out_path)
