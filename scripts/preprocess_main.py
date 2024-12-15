# preprocess_main.py
import pandas as pd
import joblib
import pickle
import argparse
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path so 'src' can be found
sys.path.append(str(Path(__file__).parent.parent))

import src.config as config

# Import necessary functions from preprocess_utils
from preprocess_utils import (
    load, create_entity_ids, sample_inference_data, time_based_split, handle_missing_values,
    encode_categoricals, scale_numeric_features
)

def run_common_preprocessing(
    transaction_path,
    identity_path,
    n_rows=None,
    target_col=config.TARGET_COL,
    id_col=config.ID_COL,
    timestamp_col=config.TIMESTAMP_COL,
    exclude_cols_initial=None,
    num_cols=None,
    cat_cols=None,
    entity_id_cols=None, # Pass entity IDs explicitly
    test_size=config.TEST_SPLIT_SIZE,
    val_size=config.VAL_SPLIT_SIZE,
    imputation_strategy=config.IMPUTATION_STRATEGY,
    cardinality_threshold=config.CARDINALITY_THRESHOLD_FOR_OHE,
    embedding_dim_factor=config.GNN_EMBEDDING_DIM_FACTOR,
    max_embedding_dim=config.GNN_MAX_EMBEDDING_DIM,
    output_data_path=config.PROCESSED_DATA_PATH,
    output_processors_path=config.PROCESSORS_PATH
):
    """
    Performs common preprocessing steps for tabular data.

    1. Load & Merge Data (using external 'preprocess')
    2. Initial Column Dropping
    3. Define Feature Columns (if not passed)
    4. Split Data (Train/Val/Test) - CRUCIAL
    5. Impute Missing Values (Fit on Train, Transform Train/Val/Test)
    6. Encode Categorical (Fit on Train, Transform Train/Val/Test) - Returns multiple versions
    7. Scale Numerical (Fit on Train, Transform Train/Val/Test)

    Returns:
        dict: A dictionary containing processed dataframes (train, val, test)
              for different model types (e.g., 'gnn', 'xgboost'),
              and the fitted processors (imputers, encoders, scaler).
    """


    print("--- Starting Common Preprocessing ---")

    # Set defaults from config if args are None
    if exclude_cols_initial is None: exclude_cols_initial = config.DEFAULT_EXCLUDED_COLUMNS
    if num_cols is None: num_cols = config.DEFAULT_NUM_COLUMNS + config.VESTA_COLUMNS
    if cat_cols is None: cat_cols = config.DEFAULT_CAT_COLUMNS
    if entity_id_cols is None: entity_id_cols = config.DEFAULT_NON_TARGET_NODE_TYPES

    print("1. Loading and Merging Data...")
    df = load(transaction_path, identity_path, n_rows) # Use your loading function



    print("2. Creating IDs for GNN non-target node types.")
    # Create entity IDs for RGCN
    df = create_entity_ids(df)

    print("3. Defining Feature Columns...")
    if num_cols is None:
        num_cols = config.DEFAULT_NUM_COLUMNS + config.VESTA_COLUMNS # Combine lists
    if cat_cols is None:
        cat_cols = config.DEFAULT_CAT_COLUMNS

    # Ensure only existing columns are kept
    num_cols = [col for col in num_cols if col in df.columns]
    cat_cols = [col for col in cat_cols if col in df.columns]
    feature_cols = num_cols + cat_cols
    print(f"   Num Cols: {len(num_cols)}")
    print(f"   Cat Cols: {len(cat_cols)}")

    # Separate features and target
    X = df[feature_cols]
    y = df[target_col]

    print("4. Splitting Data...")

    try:
        X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(
            X=X,
            y=y,
            df_full=df, 
            timestamp_col='TransactionDT',
            test_size=0.2,
            val_size=0.15
        )
    except (ValueError, KeyError, TypeError) as e:
        print(f"Error during splitting: {e}")

    # Initialize output variables
    processors = {}
    processed_data = {'train': {}, 'val': {}, 'test': {}}

    # --- 5. Impute missing values ---
    print("5. Imputing Missing Values...")
    X_train, imputer_dict = handle_missing_values(
        X_train, strategy=imputation_strategy, num_cols=num_cols, cat_cols=cat_cols, fit=True
    )
    X_val, _ = handle_missing_values(
        X_val, num_cols=num_cols, cat_cols=cat_cols, imputer_dict=imputer_dict, fit=False
    )
    X_test, _ = handle_missing_values(
        X_test, num_cols=num_cols, cat_cols=cat_cols, imputer_dict=imputer_dict, fit=False
    )
    
    processors['imputers'] = imputer_dict
    print("   Imputation complete.")


    # --- 6a. Encoding for GNN (Label Encoding) ---
    print("6a. Encoding Categoricals for GNN...")
    X_train_gnn, encoder_gnn = encode_categoricals(X_train.copy(), cat_cols, fit=True, model_type='gnn')
    X_val_gnn, _ = encode_categoricals(X_val.copy(), cat_cols, encoder=encoder_gnn, fit=False, model_type='gnn')
    X_test_gnn, _ = encode_categoricals(X_test.copy(), cat_cols, encoder=encoder_gnn, fit=False, model_type='gnn')
    processors['encoder_gnn'] = encoder_gnn
    print("   GNN Encoding complete.")

    # --- 6b. Encoding for XGBoost (OHE / Label Encoding based on cardinality) ---
    print("6b. Encoding Categoricals for XGBoost...")
    X_train_xgb, encoder_xgb = encode_categoricals(
        X_train.copy(), # Use original imputed X_train
        cat_cols,
        fit=True,
        model_type='xgboost', # Ensures OHE/Label Encoding split
        cardinality_threshold=cardinality_threshold
    )
    X_val_xgb, _ = encode_categoricals(X_val.copy(), cat_cols, encoder=encoder_xgb, fit=False, model_type='xgboost')
    X_test_xgb, _ = encode_categoricals(X_test.copy(), cat_cols, encoder=encoder_xgb, fit=False, model_type='xgboost')
    processors['encoder_xgb'] = encoder_xgb
    print("   XGBoost Encoding complete.")

    # --- 7. Scaling Numerical Features ---
    # Note: We scale the numerical columns within the already encoded dataframes.
    # The numerical columns should still have their original names.

    print("7. Scaling Numerical Features...")
    # Scale GNN data
    num_cols_in_train_gnn = [col for col in num_cols if col in X_train_gnn.columns] # Get actual num cols present
    X_train_gnn, scaler = scale_numeric_features(X_train_gnn, num_cols_in_train_gnn, fit=True)
    X_val_gnn, _ = scale_numeric_features(X_val_gnn, num_cols_in_train_gnn, scaler=scaler, fit=False)
    X_test_gnn, _ = scale_numeric_features(X_test_gnn, num_cols_in_train_gnn, scaler=scaler, fit=False)

    # Scale XGBoost data (using the same scaler fitted above)
    num_cols_in_train_xgb = [col for col in num_cols if col in X_train_xgb.columns]
    # Important: XGBoost data might have different columns if OHE was used. Scale only the original num cols.
    X_train_xgb, _ = scale_numeric_features(X_train_xgb, num_cols_in_train_xgb, scaler=scaler, fit=False) # Use existing scaler
    X_val_xgb, _ = scale_numeric_features(X_val_xgb, num_cols_in_train_xgb, scaler=scaler, fit=False)
    X_test_xgb, _ = scale_numeric_features(X_test_xgb, num_cols_in_train_xgb, scaler=scaler, fit=False)

    processors['scaler'] = scaler
    print("   Scaling complete.")

    # Store results
    processed_data['train']['gnn'] = X_train_gnn
    processed_data['val']['gnn'] = X_val_gnn
    processed_data['test']['gnn'] = X_test_gnn
    processed_data['train']['xgb'] = X_train_xgb
    processed_data['val']['xgb'] = X_val_xgb
    processed_data['test']['xgb'] = X_test_xgb

    # Add labels
    processed_data['train']['y'] = y_train
    processed_data['val']['y'] = y_val
    processed_data['test']['y'] = y_test

    # Add original df index if needed for graph construction
    processed_data['train']['index'] = X_train.index
    processed_data['val']['index'] = X_val.index
    processed_data['test']['index'] = X_test.index

    # --- 8. Create the snapshot for GNN ---
    print("8. Creating snapshot with IDs for graph construction.")
    # Get the entity ID columns (assuming they are defined in config)
    entity_id_cols = config.DEFAULT_NON_TARGET_NODE_TYPES

    # Add other essential/useful columns
    snapshot_cols = entity_id_cols + [config.ID_COL, config.TIMESTAMP_COL] # e.g., ['card_id', 'locality_id', ..., 'TransactionID', 'TransactionDT']

    # Ensure these columns actually exist in df before selecting
    snapshot_cols_present = [col for col in snapshot_cols if col in df.columns]
    if len(snapshot_cols_present) != len(snapshot_cols):
        missing_snapshot_cols = set(snapshot_cols) - set(snapshot_cols_present)
        print(f"Warning: Columns intended for snapshot not found in df: {missing_snapshot_cols}")
        # Decide how to handle this - raise error or proceed with available columns?

    # Create the optimized snapshot for GNN
    # Select only the necessary columns for the relevant rows (indexed by X.index)
    # This is needed by the GNN dataset to build edges using original IDs
    processed_data['full_df_original_ids'] = df.loc[X.index, snapshot_cols_present]

    # Add final column lists used after exclusions to processors for reference
    processors['final_num_cols'] = num_cols
    processors['final_cat_cols'] = cat_cols

    # --- Save Outputs ---
    print(f"Saving processed data to {output_data_path}")
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    with open(output_data_path, 'wb') as f:
        pickle.dump(processed_data, f)

    print(f"Saving fitted processors to {output_processors_path}")
    os.makedirs(os.path.dirname(output_processors_path), exist_ok=True)
    joblib.dump(processors, output_processors_path)

    try:
        sample_inference_data(
            df_original=df, # Pass the df *after* create_entity_ids
            test_indices=X_test.index, # Use the test indices from the split
            entity_id_cols=entity_id_cols, # Use the list passed to the function
            target_col=target_col,
            n_samples=50, # Target sample size
            card_id_col='card_id' # Ensure 'card_id' exists after create_entity_ids
            # Output paths default to config values
        )
    except Exception as e:
        # Catch potential errors during sampling (e.g., not enough data)
        print(f"Warning: Could not create inference sample data: {e}")


    print("Common preprocessing completed successfully.")
    # Return value not strictly needed if saving, but can be useful
    return processed_data, processors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run common preprocessing pipeline.")
    parser.add_argument("--trans_path", type=str, default=config.TRANSACTION_FILE_TRAIN, help="Path to transaction data.")
    parser.add_argument("--id_path", type=str, default=config.IDENTITY_FILE_TRAIN, help="Path to identity data.")
    parser.add_argument("--nrows", type=int, default=config.NUMBER_OF_ROWS, help="Number of rows to load (for testing).")
    parser.add_argument("--out_data", type=str, default=config.PROCESSED_DATA_PATH, help="Output path for processed data pickle.")
    parser.add_argument("--out_proc", type=str, default=config.PROCESSORS_PATH, help="Output path for processors joblib.")
    args = parser.parse_args()

    # Ensure output directories exist before running
    os.makedirs(os.path.dirname(args.out_data), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_proc), exist_ok=True)
      

    run_common_preprocessing(
        transaction_path=args.trans_path,
        identity_path=args.id_path,
        n_rows=args.nrows,
        output_data_path=args.out_data,
        output_processors_path= args.out_proc
        # Pass other config overrides via args if needed
    )
