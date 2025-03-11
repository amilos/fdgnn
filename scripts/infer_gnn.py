# infer_gnn.py
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd
import joblib
import argparse
import os
import sys
import time
import json
import numpy as np

# Add the parent directory to the Python path so 'src' can be found
sys.path.append(str(Path(__file__).parent.parent))

import src.config as config
# Import necessary functions from preprocess_utils
from scripts.preprocess_utils import preprocess_for_inference, load, create_entity_ids
# Import my specific GNN model class(es)
from src.gnn_model import FraudGNN
from torch_geometric.data import Data, HeteroData # Import Data

# NOTE: This script provides a framework. The actual GNN inference logic,
# especially graph construction/update for new nodes (Strategy B/C),
# requires significant implementation based on specific needs.

def infer_gnn(
    raw_trans_path,
    raw_id_path,
    processors_path,
    run_dir, # Path to the specific model run directory
    output_path="predictions/gnn_predictions.csv",
    # Add strategy argument if implementing multiple inference methods
    # inference_strategy='node_features_only', # or 'batch_graph'
    device_str='mps' # Or my preferred default
    ):
    """
    Runs inference using a trained FraudGNN model on new raw data.
    Handles loading model config and weights from the run directory.
    NOTE: Graph handling for new nodes needs specific implementation.
    """
    print(f" Starting GNN Inference for Run: {os.path.basename(run_dir)} ")

   # Determine Device 
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    # elif device_str == 'mps' and hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    #     device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

   # Load Run Configuration 
    config_path = os.path.join(run_dir, "config.json")
    model_path = os.path.join(run_dir, "model_state.pt")

    print(f"Loading run configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            hparams = json.load(f)
    except FileNotFoundError:
        print(f"Error: Run configuration file not found at {config_path}")
        return
    except json.JSONDecodeError:
         print(f"Error: Could not decode JSON from {config_path}")
         return

    print(f"Run Hyperparameters: {hparams}")
    # Extract needed params
    model_type = hparams['model_type'] # Needed to know how model expects input
    conv_type = hparams['conv_type']
    hidden_channels = hparams['hidden_channels']
    num_layers = hparams['num_layers']
    embedding_dim_other = hparams['embedding_dim_other']
    gat_heads = hparams.get('gat_heads', 4)
    num_nodes_dict_loaded = hparams.get('num_nodes_per_type')
    if num_nodes_dict_loaded is None:
        raise ValueError("'num_nodes_per_type' not found in loaded hparams (config.json). Re-run training to save it.")
 
   # Load Processors 
    print(f"Loading processors from: {processors_path}")
    try:
        processors = joblib.load(processors_path)
    except FileNotFoundError:
        print(f"Error: Processors file not found at {processors_path}")
        return
    encoder_gnn_info = processors.get('encoder_gnn')
    num_numerical_features = processors.get('num_numerical_features')
    final_num_cols = processors.get('final_num_cols') # Get final cols used
    final_cat_cols = processors.get('final_cat_cols')
    if None in [encoder_gnn_info, num_numerical_features, final_num_cols, final_cat_cols]:
         raise ValueError("Essential information missing from processors file (encoder_gnn, num_numerical_features, final columns).")
    encoder_gnn_info['num_numerical_features'] = num_numerical_features

   # Load and Preprocess New Raw Data 
    print(f"Loading new raw data from: {raw_trans_path}, {raw_id_path}")
    try:
        df_new_raw = load(raw_trans_path, raw_id_path)
        if config.ID_COL not in df_new_raw.columns:
             print(f"Warning: ID column '{config.ID_COL}' not found. Using index.")
             original_ids = pd.Series(range(len(df_new_raw)), name="index")
        else:
             original_ids = df_new_raw[config.ID_COL].copy()
        # Create entity IDs needed if building a batch graph
        df_new_raw_with_ids = create_entity_ids(df_new_raw.copy())
    except FileNotFoundError:
        print("Error: Raw inference file(s) not found.")
        return
    except Exception as e:
        print(f"Error loading/processing raw inference data: {e}")
        return

    print("Preprocessing new data features...")
    try:
        # Use final columns list from processors
        df_new_processed_features = preprocess_for_inference(
            df_new=df_new_raw_with_ids, # Use df with entity IDs created
            processor_path=processors_path,
            model_type='gnn', # Preprocess features as needed for GNN encoder
            num_cols=final_num_cols,
            cat_cols=final_cat_cols,
            exclude_cols_initial=config.DEFAULT_EXCLUDED_COLUMNS
        )
       # Convert preprocessed features to tensors 
        # Separate num/cat based on encoder_info for FeatureEncoder input
        cat_cols_encoded = list(encoder_gnn_info.get('embedding_dims', {}).keys())
        cat_cols_present = [col for col in final_cat_cols if col in df_new_processed_features.columns and col in cat_cols_encoded]
        num_cols_present = [col for col in final_num_cols if col in df_new_processed_features.columns]

        x_num_new = torch.tensor(df_new_processed_features[num_cols_present].values, dtype=torch.float32).to(device) if num_cols_present else None
        x_cat_new = torch.tensor(df_new_processed_features[cat_cols_present].values, dtype=torch.long).to(device) if cat_cols_present else None

    except Exception as e:
         print(f"Error during inference preprocessing: {e}")
         return

   # Instantiate Model 
    print("Instantiating model structure...")

    try:
        model = FraudGNN(
            node_metadata=(config.DEFAULT_NON_TARGET_NODE_TYPES + ['transaction'], []), # Provide dummy metadata structure
            num_nodes_dict=num_nodes_dict_loaded,
            encoder_info=encoder_gnn_info,
            hidden_channels=hidden_channels,
            out_channels=2,
            num_layers=num_layers,
            embedding_dim_other=embedding_dim_other,
            model_type=model_type, # Use loaded type
            conv_type=conv_type,
            heads=gat_heads
        ).to(device)
    except Exception as e:
         print(f"Error during model instantiation: {e}")
         raise

   # Load trained weights 
    print(f"Loading trained model weights from: {model_path}")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model weights file not found at {model_path}")
        return
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return
    model.eval()

   # Perform Inference 
    print("Performing inference...")
    eval_start_time = time.time()
    predictions_proba = None
    with torch.no_grad():
        try:
           # Strategy A: Node Features Only (Ignoring Graph Structure) 
            # This only uses the FeatureEncoder and the final linear layer.
            # It's fast but likely inaccurate as it ignores graph context.
            print("  Using Inference Strategy: Node Features Only (No GNN layers)")
            x_input_dict_infer = {'transaction': {'x_num': x_num_new, 'x_cat': x_cat_new}}
            # Add dummy entries for other node types if FeatureEncoder expects them
            for nt in model.node_types:
                 if nt != 'transaction' and nt not in x_input_dict_infer:
                      x_input_dict_infer[nt] = None # Or dummy tensor if needed

            x_dict_encoded = model.feature_encoder(x_input_dict_infer)
            x_transaction_encoded = x_dict_encoded.get('transaction')

            if x_transaction_encoded is None:
                 raise ValueError("FeatureEncoder did not produce transaction features.")
            if x_transaction_encoded.shape[0] > 0:
                 out_logits = model.final_layer(x_transaction_encoded)
                 predictions_proba = F.softmax(out_logits, dim=1)[:, 1] # Prob of class 1
            else:
                 predictions_proba = torch.tensor([]) # Empty result

           # Strategy B: Batch Graph Construction (Conceptual - Requires Implementation) 
            # print("  Using Inference Strategy: Batch Graph Construction (Conceptual)")
            # 1. Create entity mappings: Map entity IDs from df_new_raw_with_ids to unique indices for this batch.
            #    Handle potentially unseen entities. Need access to entities seen during training?
            # 2. Build edge_index: Create edge_index tensors connecting new transaction nodes
            #    (indices 0 to N-1) to their corresponding entity node indices within the batch graph.
            # 3. Construct Data/HeteroData: Create a PyG Data or HeteroData object for the batch.
            #    - Assign preprocessed features (x_num_new, x_cat_new) to transaction nodes.
            #    - Assign features (e.g., embeddings based on batch indices) to entity nodes.
            #    - Add the constructed edge_index.
            # 4. Run model: out = model(batch_data) # Pass the constructed batch graph
            # 5. Extract predictions: predictions_proba = F.softmax(out, dim=1)[:, 1]
            # predictions_proba = torch.rand(len(original_ids)) # Placeholder

        except Exception as e:
             print(f"Error during inference forward pass: {e}")
             # Fallback or re-raise
             predictions_proba = None # Indicate failure

    eval_duration = time.time() - eval_start_time
    print(f"Inference completed in {eval_duration:.2f}s")

   # Format and Save Output 
    if predictions_proba is None:
         print("Warning: Inference failed. No predictions generated.")
         output_df = pd.DataFrame({config.ID_COL: original_ids, config.TARGET_COL: np.nan})
    elif predictions_proba.numel() == 0:
         print("Warning: Inference produced empty output.")
         output_df = pd.DataFrame({config.ID_COL: original_ids, config.TARGET_COL: np.nan})
    else:
         output_df = pd.DataFrame({
             config.ID_COL: original_ids,
             config.TARGET_COL: predictions_proba.cpu().numpy() # Move to CPU
         })

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving predictions to: {output_path}")
    try:
        output_df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"Error saving predictions: {e}")

    print(" GNN Inference Finished ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FraudGNN inference on new data.")
    parser.add_argument("--raw_trans", type=str, default=config.TRANSACTION_FILE_TEST, help="Path to new raw transaction data.")
    parser.add_argument("--raw_id", type=str, default=config.IDENTITY_FILE_TEST, help="Path to new raw identity data.")
    parser.add_argument("--proc_path", type=str, default=config.PROCESSORS_PATH, help="Path to saved processors joblib.")
   # Use run_dir 
    parser.add_argument("--run_dir", type=str, required=False, help="Path to the specific model run directory (e.g., models/run_...).")
   # Output path 
    parser.add_argument("--out_path", type=str, default=os.path.join(config.PREDICTION_DIR, "gnn_predictions.csv"), help="Output path for predictions CSV.")
   # Optional: Add inference strategy arg 
    # parser.add_argument("--strategy", type=str, default="node_features_only", choices=["node_features_only", "batch_graph"])
    parser.add_argument("--device", type=str, default='mps', help="Device to use ('cuda', 'cpu', 'mps').")
    args = parser.parse_args()

    # Create prediction directory if it doesn't exist
    os.makedirs(config.PREDICTION_DIR, exist_ok=True)

    run_dir = "models/20250330_160352_homo_SAGE_h128_l3" 

    infer_gnn(
        raw_trans_path=args.raw_trans,
        raw_id_path=args.raw_id,
        processors_path=args.proc_path,
        run_dir=run_dir,
        output_path=args.out_path,
        # strategy=args.strategy,
        device_str=args.device
    )
