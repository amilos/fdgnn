# evaluate_gnn.py
from pathlib import Path
import torch
import torch.nn.functional as F
import pickle
import joblib
import argparse
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score
import os
import sys
import time
import json # To load config
import numpy as np # For metrics calculation if needed

# Add the parent directory to the Python path so 'src' can be found
sys.path.append(str(Path(__file__).parent.parent))

import src.config as config
from src.gnn_dataset import IeeeFraudDetectionDataset
from src.gnn_model import FraudGNN # Import the main wrapper model
# Import the conversion helper if needed for homo mode
from scripts.train_gnn import convert_to_homogeneous # Assuming it's in train_gnn or moved to utils
from torch_geometric.data import Data, HeteroData

def evaluate_gnn(
    processed_data_path,
    processors_path,
    run_dir, # Path to the specific model run directory
    device_str='mps' # Or my preferred default
    ):
    """Evaluates a trained FraudGNN model from a specific run directory on the test set."""
    print(f" Starting GNN Evaluation for Run: {os.path.basename(run_dir)} ")

   # Determine Device 
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    # elif device_str == 'mps' and hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    #     device = torch.device('mps') # Enable if MPS works
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

   # Load Run Configuration 
    config_path = os.path.join(run_dir, "config.json")
    model_path = os.path.join(run_dir, "model_state.pt") # Standard name

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
    # Extract needed params for model instantiation
    model_type = hparams['model_type']
    conv_type = hparams['conv_type']
    hidden_channels = hparams['hidden_channels']
    num_layers = hparams['num_layers']
    embedding_dim_other = hparams['embedding_dim_other']
    gat_heads = hparams.get('gat_heads', 4) # Use .get for optional params

   # Load Processors 
    print(f"Loading processors from: {processors_path}")
    try:
        processors = joblib.load(processors_path)
    except FileNotFoundError:
        print(f"Error: Processors file not found at {processors_path}")
        return
    encoder_gnn_info = processors.get('encoder_gnn')
    num_numerical_features = processors.get('num_numerical_features') # Get from processors
    if encoder_gnn_info is None: raise ValueError("GNN encoder info missing.")
    if num_numerical_features is None: raise ValueError("Num numerical features missing.")
    encoder_gnn_info['num_numerical_features'] = num_numerical_features

   # Instantiate Dataset and Load Data 
    print("Initializing GNN Dataset...")
    try:
        dataset_root = os.path.dirname(config.PROCESSED_DATA_DIR)
        dataset = IeeeFraudDetectionDataset(
            root=dataset_root,
            processed_data_path=processed_data_path,
            processors_path=processors_path,
        )
        hetero_data = dataset[0].to(device) # Load original HeteroData
    except Exception as e:
         print(f"Error initializing dataset: {e}")
         raise

    print("Graph data loaded.")
   # Store original metadata and num_nodes_dict 
    if isinstance(hetero_data, HeteroData):
        original_metadata = hetero_data.metadata()
    else:
        raise TypeError("Initial data loaded is not HeteroData!")
    num_nodes_dict = {nt: hetero_data[nt].num_nodes for nt in hetero_data.node_types if hasattr(hetero_data[nt], 'num_nodes')}

   # Conditionally convert to get homogeneous structure 
    homo_data_structure = None
    if model_type == 'homo':
        homo_data_structure = convert_to_homogeneous(hetero_data)
        if not isinstance(homo_data_structure, Data): raise TypeError(...)
        homo_data_structure = homo_data_structure.to(device) # Move structure to device

   # Pre-calculate Features (Required by the current FraudGNN forward) 
    # This mirrors the logic in train_gnn.py before the loop
    print("Pre-calculating initial node features using FeatureEncoder...")
    # Instantiate a temporary model structure just to access the encoder easily
    # Or, ideally, make FeatureEncoder callable independently if needed often
    temp_model = FraudGNN(original_metadata, num_nodes_dict, encoder_gnn_info, hidden_channels, 2, num_layers, embedding_dim_other, model_type, conv_type, gat_heads).to(device)

    with torch.no_grad():
        x_input_dict = {}
        if 'transaction' in temp_model.node_types:
             x_input_dict['transaction'] = {
                 'x_num': hetero_data['transaction'].x_num if hasattr(hetero_data['transaction'], 'x_num') else None,
                 'x_cat': hetero_data['transaction'].x_cat if hasattr(hetero_data['transaction'], 'x_cat') else None
             }
        for node_type in temp_model.node_types:
             if node_type != 'transaction' and node_type not in x_input_dict:
                  x_input_dict[node_type] = hetero_data[node_type].x if hasattr(hetero_data[node_type], 'x') else None
        # Move inputs to device
        for node_type, features in x_input_dict.items():
             if isinstance(features, dict):
                  for key, tensor in features.items():
                       if tensor is not None: features[key] = tensor.to(device)
             elif features is not None:
                  x_input_dict[node_type] = features.to(device)

        x_dict_encoded = temp_model.feature_encoder(x_input_dict)

    x_homo = None
    if model_type == 'homo':
        print("Constructing x_homo tensor from encoded features...")
        num_total_nodes = homo_data_structure.node_type.size(0)
        x_homo = torch.zeros((num_total_nodes, hidden_channels), device=device)
        node_type_errors = False
        for i, node_type in enumerate(temp_model.node_types):
             node_mask = (homo_data_structure.node_type == i)
             num_nodes_in_mask = node_mask.sum().item()
             if node_type in x_dict_encoded and x_dict_encoded[node_type] is not None:
                  features = x_dict_encoded[node_type]
                  if features.shape[0] == num_nodes_in_mask:
                       if features.shape[0] > 0: x_homo[node_mask] = features
                  else:
                       print(f"  Error: Shape mismatch for node type '{node_type}'. Encoded: {features.shape}, Mask: {num_nodes_in_mask}")
                       node_type_errors = True
             elif num_nodes_in_mask > 0:
                  print(f"  Warning: No encoded features for node type '{node_type}', using zeros.")
        if node_type_errors: raise ValueError("Shape mismatch constructing x_homo.")
        print(f"  x_homo constructed with shape: {x_homo.shape}")

    del temp_model # Remove temporary model

   # Instantiate the actual Model for evaluation 
    print("Instantiating model structure for evaluation...")
    model = FraudGNN(
        node_metadata=original_metadata,
        num_nodes_dict=num_nodes_dict,
        encoder_info=encoder_gnn_info,
        hidden_channels=hidden_channels, # From hparams
        out_channels=2,
        num_layers=num_layers,           # From hparams
        embedding_dim_other=embedding_dim_other, # From hparams
        model_type=model_type,           # From hparams
        conv_type=conv_type,             # From hparams
        heads=gat_heads                  # From hparams
    ).to(device)

   # Load trained weights 
    print(f"Loading trained model weights from: {model_path}")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model weights file not found at {model_path}")
        return
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        print("Ensure the model architecture parameters match the trained model.")
        return
    model.eval() # Set model to evaluation mode

   # Perform Inference 
    print("Performing inference on the graph...")
    eval_start_time = time.time()
    with torch.no_grad():
        try:
            # Pass pre-calculated features/structure
            if model_type == 'hetero':
                 out = model(hetero_data, x_dict_encoded=x_dict_encoded)
            else: # homo
                 out = model(homo_data_structure, x_homo=x_homo) # Pass structure and features

        except Exception as e:
             print(f"Error during forward pass for evaluation: {e}")
             return
    eval_duration = time.time() - eval_start_time
    print(f"Inference completed in {eval_duration:.2f}s")

   # Extract Test Predictions and Labels 
    # Use original hetero_data for masks and labels
    test_mask = hetero_data['transaction'].test_mask
    target = hetero_data['transaction'].y
    if test_mask.sum() == 0:
         print("Error: No nodes found in the test mask. Cannot evaluate.")
         return

    test_logits = out[test_mask]
    test_probs = F.softmax(test_logits, dim=1)[:, 1] # Prob of class 1 (Fraud)
    test_pred_class = (test_probs > 0.5).int()
    y_test = target[test_mask]

    # Move results to CPU for scikit-learn metrics
    y_test_np = y_test.cpu().numpy()
    test_probs_np = test_probs.cpu().numpy()
    test_pred_class_np = test_pred_class.cpu().numpy()

   # Calculate Metrics 
    print("Calculating evaluation metrics...")
    try:
        if len(np.unique(y_test_np)) > 1:
            auc = roc_auc_score(y_test_np, test_probs_np)
            ap = average_precision_score(y_test_np, test_probs_np)
        else:
            print("Warning: Only one class present in test set. AUC/AP cannot be calculated reliably.")
            auc = float('nan')
            ap = float('nan')

        report = classification_report(y_test_np, test_pred_class_np, target_names=['Not Fraud', 'Fraud'], zero_division=0)
        cm = confusion_matrix(y_test_np, test_pred_class_np)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return

   # Print Results 
    print("\n GNN Evaluation Results ")
    print(f"Run ID: {os.path.basename(run_dir)}")
    print(f"Model Type: {model_type}, Conv Type: {conv_type}")
    print(f"Test AUC: {auc:.4f}")
    print(f"Test Average Precision (AP): {ap:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    if cm.shape == (2, 2):
        print(f"   TN: {cm[0][0]:<6}  FP: {cm[0][1]:<6}")
        print(f"   FN: {cm[1][0]:<6}  TP: {cm[1][1]:<6}")
    else:
         print(cm)
    print("----------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained FraudGNN model from a run directory.")
    parser.add_argument("--data_path", type=str, default=config.PROCESSED_DATA_PATH, help="Path to processed data pickle (used by dataset).")
    parser.add_argument("--proc_path", type=str, default=config.PROCESSORS_PATH, help="Path to processors joblib (used by dataset).")
   # Use run_dir 
    parser.add_argument("--run_dir", type=str, help="Path to the specific model run directory (e.g., models/run_...).")
   # Architecture args removed - loaded from config.json 
    parser.add_argument("--device", type=str, default='mps', help="Device to use ('cuda', 'cpu', 'mps').") # Keep device arg
    args = parser.parse_args()

    run_dir = "models/20250330_162353_homo_SAGE_h128_l3"    
    run_dir = "models/20250330_162733_hetero_RGCN_h128_l2"
    run_dir = "models/20250330_163731_hetero_SAGE_h32_l1"
    
    evaluate_gnn(
        processed_data_path=args.data_path,
        processors_path=args.proc_path,
        run_dir=run_dir,
        device_str=args.device
    )
