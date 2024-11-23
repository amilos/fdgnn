import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import joblib
import sys
from pathlib import Path

# Add the parent directory to the Python path so 'src' can be found
sys.path.append(str(Path(__file__).parent.parent))

import src.config as config # Import config for constants needed within functions

def create_device_profile_id(row):
    cols = ["id_30", "id_31", "id_32", "id_33", "DeviceType", "DeviceInfo"]
    values = [str(row[c]) if pd.notnull(row[c]) else 'nan' for c in cols]
    if values.count('nan') >= 3:  
        return 'missing_device'
    else:
        return '_'.join(values)
    
def create_network_profile_id(row):
    cols = ["id_17","id_19", "id_20"]
    values = [str(row[c]) if pd.notnull(row[c]) else 'nan' for c in cols]
    if values.count('nan') >= 2:  
        return 'missing_network'
    else:
        return '_'.join(values)    

def create_locality_id(row):
    cols = ["addr1", "addr2"]
    values = [str(row[c]) if pd.notnull(row[c]) else 'nan' for c in cols]
    if values.count('nan') >= 2:  
        return 'missing_locality'
    else:
        return '_'.join(values)
    
def create_card_id(row):
    cols = ["card1", "addr1", "card2", "card5", "card6"]
    transaction_day = row['TransactionDT'] / (3600 * 24)
    d1n = int((row['D1'] - transaction_day) if pd.notnull(row['D1']) else -9999)
    values = [str(row[c]) if pd.notnull(row[c]) else 'nan' for c in cols] + [str(d1n)]
    return '_'.join(values)

def create_entity_ids(df):
    """Creates combined entity IDs for GNN nodes."""
    print("Creating entity IDs...")

    df = df.copy()
    # Create a new column for device profile ID
    df['device_profile_id'] = df.apply(create_device_profile_id, axis=1)
    # Create a new column for network profile ID
    df['network_profile_id'] = df.apply(create_network_profile_id, axis=1)
    # Create a new column for locality ID
    df['locality_id'] = df.apply(create_locality_id, axis=1)
    # Create a new column for card ID
    df['card_id'] = df.apply(create_card_id, axis=1)

    print("Entity IDs created.")
    return df

def time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    df_full: pd.DataFrame, # DataFrame containing the timestamp column and aligned index with X, y
    timestamp_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Splits features (X) and target (y) into train, validation, and test sets
    based on a timestamp column present in df_full.

    Ensures that train data comes chronologically before validation data,
    which comes before test data.

    Args:
        X (pd.DataFrame): DataFrame of features.
        y (pd.Series): Series of target variable.
        df_full (pd.DataFrame): The original DataFrame from which X and y were derived,
                                MUST contain the timestamp_col and have an index
                                that aligns with X and y.
        timestamp_col (str): The name of the column containing the timestamp information
                             in df_full.
        test_size (float): Proportion of the dataset to include in the test split
                           (e.g., 0.2 for 20%).
        val_size (float): Proportion of the dataset to include in the validation split
                          (e.g., 0.1 for 10%).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
            A tuple containing (X_train, X_val, X_test, y_train, y_val, y_test).

    Raises:
        ValueError: If timestamp column is not found, if sizes are invalid,
                    or if X, y, and df_full indices don't align properly.
        KeyError: If timestamp_col is not in df_full.
    """
    print("Splitting Data (Time-Based)...")

    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise TypeError("X must be a pandas DataFrame and y must be a pandas Series.")
    if not X.index.equals(y.index):
        raise ValueError("Indices of X and y do not match.")
    if timestamp_col not in df_full.columns:
        raise KeyError(f"Timestamp column '{timestamp_col}' not found in df_full.")
    if not df_full.index.isin(X.index).all() or not X.index.isin(df_full.index).all():
         # Check if indices are compatible (might not be identical if df_full has more rows)
         if not X.index.isin(df_full.index).all():
              raise ValueError("Index of X contains values not present in df_full's index.")
         # It's okay if df_full has more indices than X, as long as all X indices are in df_full

    # --- Time-Based Split ---
    # Select timestamps corresponding to the rows in X/y and sort
    temp_df_for_split = df_full.loc[X.index, [timestamp_col]].copy()
    temp_df_for_split = temp_df_for_split.sort_values(by=timestamp_col)

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

    # Create the final splits using the selected indices from X and y
    X_train = X.loc[train_indices]
    y_train = y.loc[train_indices]
    X_val = X.loc[val_indices]
    y_val = y.loc[val_indices]
    X_test = X.loc[test_indices]
    y_test = y.loc[test_indices]

    # --- End Time-Based Split ---

    print(f"   Train shape: {X_train.shape} (Indices from {train_indices.min()} to {train_indices.max()})")
    print(f"   Val shape: {X_val.shape} (Indices from {val_indices.min()} to {val_indices.max()})")
    print(f"   Test shape: {X_test.shape} (Indices from {test_indices.min()} to {test_indices.max()})")


    return X_train, X_val, X_test, y_train, y_val, y_test

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

    # Check for missing columns before proceeding
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
        if imputer_dict is None: 
            raise ValueError("imputer_dict needed for fit=False.")
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

def encode_categoricals(df, cat_cols, encoder=None, fit=True, model_type='xgboost', max_embedding_dim=32, cardinality_threshold=10) -> tuple[pd.DataFrame, dict]:
    """
    Encode categorical features based on cardinality and target model type.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        cat_cols (list): List of categorical columns to encode.
        encoder (dict, optional): Existing encoder dictionary from a previous fit. Defaults to None.
        fit (bool): Whether to fit a new encoder (True) or use existing one (False). Defaults to True.
        model_type (str): 'xgboost' or 'gnn' - determines encoding strategy. Defaults to 'xgboost'.
        max_embedding_dim (int): Maximum embedding dimension for GNNs. Defaults to 50.
        cardinality_threshold (int): Max cardinality for using one-hot encoding in XGBoost. Defaults to 10.

    Returns:
        tuple:
            - pd.DataFrame: DataFrame with encoded features.
            - dict: Dictionary containing fitted encoders, calculated embedding dimensions (for GNN),
                    original categories count, and lists of low/high cardinality columns.
    """

    print(f"   Encoding categoricals for {model_type} (fit={fit})...")

    df = df.copy()

    if fit:
        # Initialize encoder structure if fitting
        encoder = {'ohe': None, 'embedding_dims': {}, 'categories': {}, 'high_card_cols': [], 'low_card_cols': [], 'label_encoders': {}} # Added label_encoders

        # Calculate cardinality for each categorical column
        encoder['categories'] = {col: df[col].astype('category').nunique() for col in cat_cols} # Ensure consistent counting

        # Separate columns based on cardinality
        encoder['low_card_cols'] = [col for col in cat_cols if encoder['categories'][col] <= cardinality_threshold]
        encoder['high_card_cols'] = [col for col in cat_cols if encoder['categories'][col] > cardinality_threshold]

        # --- GNN Specific Fitting ---
        if model_type == 'gnn':
            for col in cat_cols:
                cardinality = encoder['categories'][col]
                # Rule of thumb: embedding_dim = min(max_dim, embedding_factor * sqrt(cardinality))
                # Using a slightly more common heuristic, often just based on cardinality directly or log
                # embedding_dim = min(max_embedding_dim, int(embedding_dim_factor * math.sqrt(cardinality))) # Original
                embedding_dim = min(max_embedding_dim, max(2, int(cardinality / 2))) # Alternative heuristic: half cardinality, min 2, max 50
                # embedding_dim = min(max_embedding_dim, max(2, int(math.log2(cardinality) * 4))) # Another heuristic using log
                # Choose one heuristic or make it configurable
                encoder['embedding_dims'][col] = embedding_dim

                # Store category mapping for consistent label encoding
                encoder['label_encoders'][col] = {k: i for i, k in enumerate(df[col].astype('category').cat.categories)}


        # --- XGBoost Specific Fitting ---
        elif model_type == 'xgboost':
            # Fit OneHotEncoder ONLY for low cardinality features for XGBoost
            if encoder['low_card_cols']:
                # Use sparse_output=False for newer scikit-learn versions
                encoder['ohe'] = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                encoder['ohe'].fit(df[encoder['low_card_cols']])

            # Store category mapping for consistent label encoding for high cardinality features
            for col in encoder['high_card_cols']:
                 encoder['label_encoders'][col] = {k: i for i, k in enumerate(df[col].astype('category').cat.categories)}

    # --- Transformation ---
    if encoder is None:
         raise ValueError("Encoder must be provided or fit must be True.")

    if model_type == 'xgboost':
        # Apply OHE for low cardinality
        if encoder['low_card_cols'] and encoder.get('ohe'): # Check if OHE exists
            try:
                encoded = encoder['ohe'].transform(df[encoder['low_card_cols']])
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=encoder['ohe'].get_feature_names_out(encoder['low_card_cols']),
                    index=df.index
                )
                # Drop original low card cols and concat OHE cols
                df = df.drop(columns=encoder['low_card_cols'])
                df = pd.concat([df, encoded_df], axis=1)
            except ValueError as e:
                 print(f"Warning: Error applying OHE (potentially unseen categories if handle_unknown='error'): {e}")
                 # Handle cases where columns might be missing in the df passed during transform
                 pass


        # Apply Label Encoding for high cardinality
        if encoder['high_card_cols']:
            for col in encoder['high_card_cols']:
                if col in df.columns: # Check if column exists
                    # Map known categories, assign -1 or another value to unknowns
                    mapping = encoder['label_encoders'].get(col, {})
                    df[col] = df[col].map(mapping).fillna(-1).astype(int) # Use stored mapping
                else:
                     print(f"Warning: Column '{col}' not found during XGBoost high-cardinality encoding.")


    elif model_type == 'gnn':
        # Apply Label Encoding (integer codes) for ALL categorical features for GNN embedding lookup
        for col in cat_cols:
             if col in df.columns: # Check if column exists
                # Map known categories, assign -1 or another value to unknowns
                mapping = encoder['label_encoders'].get(col, {})
                df[col] = df[col].map(mapping).fillna(-1).astype(int) # Use stored mapping
             else:
                 print(f"Warning: Column '{col}' not found during GNN encoding.")


    return df, encoder

def scale_numeric_features(df: pd.DataFrame, num_cols: list, scaler: StandardScaler = None, fit: bool = True) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Scales numeric features using StandardScaler.

    Assumes the same scaling is appropriate for both XGBoost and GNNs,
    which is generally a safe default (critical for GNNs, often neutral
    or slightly beneficial for XGBoost).

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        num_cols (list): List of numerical column names to scale.
        scaler (StandardScaler, optional): A pre-fitted StandardScaler instance.
                                           Required if fit=False. Defaults to None.
        fit (bool): If True, fit a new scaler and transform.
                    If False, use the provided scaler to transform. Defaults to True.

    Returns:
        tuple[pd.DataFrame, StandardScaler]:
            - DataFrame with scaled numeric features.
            - The fitted or provided StandardScaler instance.

    Raises:
        ValueError: If fit=False and scaler is None.
        NotFittedError: If fit=False and the provided scaler is not fitted.
        KeyError: If columns in num_cols are not found in df during transform.
    """

    df = df.copy()

    # Check for missing columns before proceeding
    missing_cols = [col for col in num_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in DataFrame: {missing_cols}")

    if fit:
        scaler = StandardScaler()
        # Fit and transform in one step
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        if scaler is None:
            raise ValueError("A fitted scaler must be provided when fit=False.")
        try:
            # Check if scaler is fitted (will raise NotFittedError if not)
            check_is_fitted(scaler)

            # Transform using the provided scaler
            df[num_cols] = scaler.transform(df[num_cols])
        except NotFittedError as e:
            raise NotFittedError("The provided scaler is not fitted.") from e
        except KeyError as e:
             # This might happen if df during transform has different columns
             # than the df used during fit, although the initial check helps.
             raise KeyError(f"Columns mismatch during transform: {e}")

    return df, scaler

def load(transaction_path, identity_path, n_rows=10000):
    """Loads and merges transaction and identity data."""
    print(f"Loading data... (nrows={n_rows})")
    offset = 13000 # first 13000 rows have a lot more missing values in identity
    try:
        if n_rows is None:
            df_trans = pd.read_csv(transaction_path, low_memory=False)
            df_id = pd.read_csv(identity_path, low_memory=False)
        else:
            df_trans = pd.read_csv(transaction_path, nrows=int(offset + n_rows), low_memory=False)
            df_id = pd.read_csv(identity_path, low_memory=False)
        # Use TransactionID from config
        df = pd.merge(df_trans, df_id, on=config.ID_COL, how='left')
        # Sort by Transactio timestamp
        df = df.sort_values("TransactionDT")
        # Keep only the last n_rows
        df = df.tail(n_rows)

        print(f"Data loaded and merged. Shape: {df.shape}")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure files exist at specified paths.")
        raise
    return df

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
