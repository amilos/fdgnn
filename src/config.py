# config.py
"""
Central configuration file for column names, paths, and constants.
"""
import os

# --- File Paths (adjust as needed) ---

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get parent of src dir
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
PREDICTION_DIR = os.path.join(PROJECT_ROOT, "predictions") # Added for outputs

TRANSACTION_FILE_TRAIN = os.path.join(RAW_DATA_DIR, "train_transaction.csv")
IDENTITY_FILE_TRAIN = os.path.join(RAW_DATA_DIR, "train_identity.csv")

TRANSACTION_FILE_TEST = os.path.join(RAW_DATA_DIR, "test_transaction.csv")
IDENTITY_FILE_TEST = os.path.join(RAW_DATA_DIR, "test_identity.csv")

PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_data.pkl")
PROCESSORS_PATH = os.path.join(PROCESSED_DATA_DIR, "processors.joblib")
GRAPH_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "graph_data.pt") # Optional path for PyG data

XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.json")
GNN_MODEL_PATH = os.path.join(MODEL_DIR, "gnn_model.pt")


# Paths for the sampled inference demo files
INFERENCE_SAMPLE_TRANS_PATH = os.path.join(RAW_DATA_DIR, "test_transaction.csv")
INFERENCE_SAMPLE_ID_PATH = os.path.join(RAW_DATA_DIR, "test_identity.csv")


# These are needed to split the sampled data back into original format
ORIGINAL_TRANSACTION_COLS = [ 
    'TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'ProductCD',
    'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2',
    'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
    'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
    # Add Vesta columns if they are considered part of transaction raw data
] + [f'V{i}' for i in range(1, 340)] # Example Vesta cols

ORIGINAL_IDENTITY_COLS = [ 
    'TransactionID', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06',
    'id_07', 'id_08', 'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14',
    'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22',
    'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30',
    'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',
    'DeviceType', 'DeviceInfo'
]




# --- Column Definitions ---
TARGET_COL = 'isFraud'
ID_COL = 'TransactionID'
TIMESTAMP_COL = 'TransactionDT'

# --- Columns to Exclude ---
DEFAULT_EXCLUDED_COLUMNS = [
    # idetifier of target node, target label, timestamp of transaction
    "TransactionID", "isFraud", "TransactionDT", 
    # Not selective enough 
    'P_emaildomain', 'R_emaildomain', "id_17",
    # Too many missing values 
    "dist2", "D6", "D7", "D8", "D9", "D12", "D13", "D14",    
    "id_03", "id_04", "id_07", "id_08", "id_09", "id_10", "id_18", 
    "id_21", "id_22", "id_23", "id_24", "id_25", "id_26"
]


GNN_EXCLUDED_COLUMNS = [
    # Reduntant with engineered card_id
    "D1", "card1", "card2", "card3", "card4", "card5", "card6",
    # Reduntant with engineered network_id
    "id_19", "id_20",
    # Reduntant with engineered device_profile_id
    "id_30", "id_31", "id_32", "id_33", "DeviceType", "DeviceInfo",
    # Redundant with engineered locality_id
    "addr1", "addr2"
]  

# --- Feature Columns ---
DEFAULT_CAT_COLUMNS = [
    "id_13", # Cardinality 32, recommended embedding dim = 8
    "id_14", # Timezone with cardinality 49, recommended embedding dim = 8

    # Cardinality 2
    "M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9",    
    "id_12", "id_16", "id_27", "id_28", "id_29", "id_35", "id_36", "id_37", "id_38",

    # Cardinality 3
    "ProductCD", "M4", "id_15", "id_34" 

]

DEFAULT_NUM_COLUMNS = [
    "TransactionAmt", # Transaction amount
    "dist1",
    # Counts
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", 
    # Time deltas
    "D2", "D3", "D4", "D5", "D10", "D11", "D15",
    # ID ranks
    "id_01", "id_02", "id_05", "id_06", "id_11"
]

VESTA_COLUMNS = [
    # V1-V11
    "V1", "V3", "V4", "V6", "V8", "V11",
    # V13-V30
    "V13", "V14", "V17", "V20", "V23", "V26", "V27", "V30",
    # V36-V48
    "V36", "V37", "V40", "V41", "V44", "V47", "V48",
    # V54-V70
    "V54", "V56", "V59", "V62", "V65", "V67", "V68", "V70",
    # V76-V91
    "V76", "V78", "V80", "V82", "V86", "V88", "V89", "V91",
    # V96-V104 
    "V96", "V98", "V99", "V104",
    # V284-V297 
    "V284", "V285", "V286", "V291", "V294", "V297",
    # V303-V320 
    "V303", "V305", "V307", "V309", "V310", "V320",
    # V281-V314 
    "V281", "V283", "V289", "V296", "V301", "V314"
]


vesta_engineered_columns = [f"V{i}" for i in range(1, 340)]

# Compute the excluded columns as the difference
VESTA_EXCLUDED_COLUMNS = [col for col in vesta_engineered_columns if col not in VESTA_COLUMNS]

# GNN Node Types (columns used to define non-target nodes)
# These columns MUST be created in `create_entity_ids` or exist in the raw data
DEFAULT_NON_TARGET_NODE_TYPES = [
    "card_id",
    "device_profile_id",
    "network_profile_id",
    "locality_id",
]


# --- Preprocessing Parameters ---
NUMBER_OF_ROWS = 100000
TEST_SPLIT_SIZE = 0.20
VAL_SPLIT_SIZE = 0.15 # Note: Train size = 1.0 - test_size - val_size
IMPUTATION_STRATEGY = "median"
CARDINALITY_THRESHOLD_FOR_OHE = 10 # For XGBoost encoding

# --- Model Hyperparameters ---
# XGBoost
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'min_child_weight': 1,
    'gamma': 0.1,
    'lambda': 1,
    'alpha': 0,
    'seed': 42
}
XGB_NUM_BOOST_ROUND = 500
XGB_EARLY_STOPPING_ROUNDS = 50

# GNN

GNN_LEARNING_RATE = 0.001
GNN_EPOCHS = 500
GNN_HIDDEN_DIM = 128 # Increased from 128
GNN_EMBEDDING_DIM_FACTOR = 4
GNN_MAX_EMBEDDING_DIM = 64
GNN_PATIENCE = 10
GNN_NUM_LAYERS = 3 # Keeping layers at 2 for now
GNN_EMB_DIM_OTHER = 64 # Increased from 32


GNN_SCHEDULER_FACTOR = 0.5 # Factor to reduce LR by
GNN_SCHEDULER_PATIENCE = 5  # Epochs to wait for improvement before reducing LR
GNN_SCHEDULER_MIN_LR = 1e-6 # Minimum learning rate
