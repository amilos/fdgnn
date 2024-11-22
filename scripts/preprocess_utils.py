# preprocess_utils.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import joblib


import config # Import config for constants needed within functions


def load(transaction_path, identity_path, n_rows=None):
    """Loads and merges transaction and identity data."""
    print(f"Loading data... (nrows={n_rows})")
    try:
        df_trans = pd.read_csv(transaction_path, nrows=n_rows)
        df_id = pd.read_csv(identity_path, nrows=n_rows)
        # Use TransactionID from config
        df = pd.merge(df_trans, df_id, on=config.ID_COL, how='left')
        print(f"Data loaded and merged. Shape: {df.shape}")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure files exist at specified paths.")
        raise
    return df

def create_entity_ids(df):
    """Creates combined entity IDs for GNN nodes."""
    print("Creating entity IDs...")

    # TODO: ADD ENTITY ID CREATION LOGIC HERE ---
    # Ensure the column names match config.DEFAULT_NON_TARGET_NODE_TYPES

  
    print("Entity IDs created.")
    return df

def time_based_split(
    df_full: pd.DataFrame,
    timestamp_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    """
    Performs time-based split based on a timestamp column.

    Args:
        df_full (pd.DataFrame): DataFrame containing the timestamp column.
        timestamp_col (str): Name of the timestamp column.
        test_size (float): Proportion for the test set.
        val_size (float): Proportion for the validation set.

    Returns:
        tuple[pd.Index, pd.Index, pd.Index]: Train, validation, and test indices.
    """
    print("Splitting Data (Time-Based)...")
    if timestamp_col not in df_full.columns:
        raise KeyError(f"Timestamp column '{timestamp_col}' not found in df_full.")

    # Select timestamps and sort
    temp_df_for_split = df_full[[timestamp_col]].sort_values(by=timestamp_col)

    # Calculate split sizes
    n_samples = len(temp_df_for_split)
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * val_size)
    n_train = n_samples - n_val - n_test

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Calculated split sizes are invalid (Train: {n_train}, Val: {n_val}, Test: {n_test}). "
            f"Check val_size ({val_size}) and test_size ({test_size}). They might sum to >= 1.0"
        )

    # Get the indices corresponding to the sorted order
    train_indices = temp_df_for_split.index[:n_train]
    val_indices = temp_df_for_split.index[n_train : n_train + n_val]
    test_indices = temp_df_for_split.index[n_train + n_val :]

    print(f"   Train size: {len(train_indices)}, Val size: {len(val_indices)}, Test size: {len(test_indices)}")
    return train_indices, val_indices, test_indices


# --- PASTE YOUR REVIEWED/FINAL VERSIONS OF ---
# handle_missing_values(...)
# encode_categoricals(...)
# scale_numeric_features(...)
# preprocess_for_inference(...)
# --- HERE, ensuring they import necessary libraries ---

# Example placeholder for handle_missing_values (replace with your actual code)
# This version assumes you are EXCLUDING fully missing columns in preprocess_main.py
# So it doesn't need the complex handling internally.
def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
    num_cols: list = None,
    cat_cols: list = None,
    imputer_dict: dict = None,
    fit: bool = True
) -> tuple[pd.DataFrame, dict]:
    """Handles missing values using SimpleImputer."""
    df = df.copy()
    if num_cols is None: num_cols = []
    if cat_cols is None: cat_cols = []

    all_cols = num_cols + cat_cols
    missing_in_df = [col for col in all_cols if col not in df.columns]
    if missing_in_df:
        raise KeyError(f"Columns not found in DataFrame for imputation: {missing_in_df}")

    CAT_NAN_FILL_VALUE = 'missing' # Consistent fill value

    if fit:
        imputer_dict = {'num': None, 'cat': None}
        # Numeric imputation
        if num_cols:
            num_imputer = SimpleImputer(strategy=strategy)
            # Check for columns with data before fitting
            cols_to_fit_num = [col for col in num_cols if not df[col].isnull().all()]
            if cols_to_fit_num:
                 print(f"   Fitting numerical imputer on {len(cols_to_fit_num)} cols.")
                 df[cols_to_fit_num] = num_imputer.fit_transform(df[cols_to_fit_num])
                 imputer_dict['num'] = num_imputer
            else:
                 print("   No numerical columns with data to fit imputer.")
            # Fill any remaining fully NaN num cols (shouldn't happen if excluded earlier)
            fully_nan_num = [col for col in num_cols if col not in cols_to_fit_num]
            if fully_nan_num:
                 print(f"   Warning: Filling fully NaN numerical columns {fully_nan_num} with 0.0 (should have been excluded).")
                 df[fully_nan_num] = df[fully_nan_num].fillna(0.0)
        # Categorical imputation
        if cat_cols:
            cat_imputer = SimpleImputer(strategy='constant', fill_value=CAT_NAN_FILL_VALUE)
            cols_to_fit_cat = [col for col in cat_cols if not df[col].isnull().all()]
            if cols_to_fit_cat:
                 print(f"   Fitting categorical imputer on {len(cols_to_fit_cat)} cols.")
                 df[cols_to_fit_cat] = df[cols_to_fit_cat].astype(object) # Ensure correct type
                 df[cols_to_fit_cat] = cat_imputer.fit_transform(df[cols_to_fit_cat])
                 imputer_dict['cat'] = cat_imputer
            else:
                 print("   No categorical columns with data to fit imputer.")
            fully_nan_cat = [col for col in cat_cols if col not in cols_to_fit_cat]
            if fully_nan_cat:
                 print(f"   Warning: Filling fully NaN categorical columns {fully_nan_cat} with '{CAT_NAN_FILL_VALUE}' (should have been excluded).")
                 df[fully_nan_cat] = df[fully_nan_cat].astype(object).fillna(CAT_NAN_FILL_VALUE)

    else: # Transform only
        if imputer_dict is None: raise ValueError("imputer_dict needed for fit=False.")
        num_imputer = imputer_dict.get('num')
        cat_imputer = imputer_dict.get('cat')

        if num_imputer is not None and num_cols:
            cols_to_transform_num = [col for col in num_cols if col in df.columns]
            if cols_to_transform_num:
                check_is_fitted(num_imputer)
                df[cols_to_transform_num] = num_imputer.transform(df[cols_to_transform_num])
        elif num_cols: # Handle case where imputer wasn't fitted but cols exist
             print(f"   Warning: Numerical columns {num_cols} need imputation, but no fitted imputer found. Filling NaNs with 0.0.")
             df[num_cols] = df[num_cols].fillna(0.0)


        if cat_imputer is not None and cat_cols:
            cols_to_transform_cat = [col for col in cat_cols if col in df.columns]
            if cols_to_transform_cat:
                check_is_fitted(cat_imputer)
                df[cols_to_transform_cat] = df[cols_to_transform_cat].astype(object)
                df[cols_to_transform_cat] = cat_imputer.transform(df[cols_to_transform_cat])
        elif cat_cols: # Handle case where imputer wasn't fitted but cols exist
             print(f"   Warning: Categorical columns {cat_cols} need imputation, but no fitted imputer found. Filling NaNs with '{CAT_NAN_FILL_VALUE}'.")
             df[cat_cols] = df[cat_cols].astype(object).fillna(CAT_NAN_FILL_VALUE)

    return df, imputer_dict


# Example placeholder for encode_categoricals (replace with your actual code)
def encode_categoricals(
    df: pd.DataFrame,
    cat_cols: list,
    encoder: dict = None,
    fit: bool = True,
    model_type: str = 'xgboost',
    **kwargs # To catch extra args like cardinality_threshold etc.
) -> tuple[pd.DataFrame, dict]:
    """Encodes categorical features based on model type."""
    print(f"   Encoding categoricals for {model_type} (fit={fit})...")
    df = df.copy()
    cat_cols = [col for col in cat_cols if col in df.columns] # Use only existing cols

    if fit:
        # --- PASTE YOUR FITTING LOGIC HERE ---
        # Needs to handle 'gnn' (LabelEncoding + store embedding dims)
        # and 'xgboost' (OHE for low card, LabelEncoding for high card)
        # Store fitted encoders (LabelEncoder maps, OHE object) and embedding dims in 'encoder' dict
        encoder = {'type': model_type, 'label_encoders': {}, 'ohe': None, 'embedding_dims': {}, 'low_card_cols': [], 'high_card_cols': []} # Example structure
        print("   Placeholder: Fit logic for encode_categoricals needs implementation.")
    else:
        if encoder is None: raise ValueError("Encoder dict needed for fit=False.")
        # --- PASTE YOUR TRANSFORM LOGIC HERE ---
        # Apply transformations based on the 'encoder' dict and model_type
        print("   Placeholder: Transform logic for encode_categoricals needs implementation.")

    return df, encoder


# Example placeholder for scale_numeric_features (replace with your actual code)
def scale_numeric_features(
    df: pd.DataFrame,
    num_cols: list,
    scaler: StandardScaler = None, # Explicitly type hint
    fit: bool = True
) -> tuple[pd.DataFrame, StandardScaler]:
    """Scales numeric features using StandardScaler."""
    df = df.copy()
    num_cols = [col for col in num_cols if col in df.columns] # Use only existing cols

    if not num_cols:
        print("   No numerical columns found to scale.")
        return df, scaler # Return unchanged df and potentially None scaler

    if fit:
        print(f"   Fitting StandardScaler on {len(num_cols)} numerical columns.")
        scaler = StandardScaler()
        # --- PASTE YOUR FIT_TRANSFORM LOGIC HERE ---
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        if scaler is None: raise ValueError("Scaler must be provided when fit=False.")
        print(f"   Applying StandardScaler transform to {len(num_cols)} numerical columns.")
        # --- PASTE YOUR TRANSFORM LOGIC HERE ---
        check_is_fitted(scaler)
        df[num_cols] = scaler.transform(df[num_cols])

    return df, scaler


# Example placeholder for preprocess_for_inference (replace with your actual code)
def preprocess_for_inference(
    df_new: pd.DataFrame,
    processor_path: str,
    model_type: str,
    num_cols: list,
    cat_cols: list,
    exclude_cols_initial: list
) -> pd.DataFrame:
     """Applies saved preprocessing steps to new data."""
     print(f"Preprocessing new data for {model_type} inference...")
     df_processed = df_new.copy()

     # 1. Load Processors
     try:
         processors = joblib.load(processor_path)
     except FileNotFoundError:
         print(f"Error: Processors file not found at {processor_path}")
         raise

     # 2. Initial Column Dropping
     cols_to_drop = [col for col in exclude_cols_initial if col in df_processed.columns]
     df_processed = df_processed.drop(columns=cols_to_drop)
     print(f"   Dropped initial columns: {cols_to_drop}")

     # Ensure required columns exist after dropping
     current_num_cols = [col for col in num_cols if col in df_processed.columns]
     current_cat_cols = [col for col in cat_cols if col in df_processed.columns]
     required_cols = set(num_cols + cat_cols)
     available_cols = set(df_processed.columns)
     if not required_cols.issubset(available_cols):
          missing = required_cols - available_cols
          # Decide how to handle - maybe only use available required cols?
          print(f"Warning: Required feature columns missing after initial drop: {missing}")
          # Filter lists to only available columns expected by processors
          current_num_cols = [col for col in num_cols if col in available_cols]
          current_cat_cols = [col for col in cat_cols if col in available_cols]


     # 3. Impute (fit=False)
     print("   Applying imputation...")
     imputer_dict = processors.get('imputers')
     if imputer_dict:
         df_processed, _ = handle_missing_values(
             df_processed, num_cols=current_num_cols, cat_cols=current_cat_cols,
             imputer_dict=imputer_dict, fit=False
         )
     else:
         print("   Warning: Imputer dictionary not found in processors.")

     # 4. Encode (fit=False, correct model_type)
     print(f"   Applying {model_type} categorical encoding...")
     encoder_key = 'encoder_xgb' if model_type == 'xgboost' else 'encoder_gnn'
     encoder = processors.get(encoder_key)
     if encoder:
         df_processed, _ = encode_categoricals(
             df_processed, current_cat_cols, encoder=encoder, fit=False, model_type=model_type
         )
     else:
         print(f"   Warning: Encoder '{encoder_key}' not found in processors.")


     # 5. Scale (fit=False)
     print("   Applying numerical scaling...")
     scaler = processors.get('scaler')
     # Get numerical columns *after* potential encoding changes (e.g., OHE)
     # This assumes original numerical column names are preserved or identifiable
     num_cols_after_encoding = [col for col in current_num_cols if col in df_processed.columns]
     if scaler and num_cols_after_encoding:
         df_processed, _ = scale_numeric_features(
             df_processed, num_cols_after_encoding, scaler=scaler, fit=False
         )
     elif num_cols_after_encoding:
          print("   Warning: Scaler not found in processors, but numerical columns exist.")


     print("Inference preprocessing complete.")
     return df_processed
