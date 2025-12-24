# evaluate_gnn.py
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import joblib
import argparse
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score
import os
import time
import sys

sys.path.append(str(Path(__file__).parent.parent))

import src.config as config

from src.gnn_dataset import IeeeFraudDetectionDataset
# Import your specific GNN model class(es)
from src.gnn_model import FraudGNN # Import the main wrapper model

def evaluate_gnn(
    processed_data_path,
    processors_path,
    model_path,
    # --- Model architecture parameters (MUST match training) ---
    model_type='hetero',
    conv_type='SAGE',
    hidden_channels=config.GNN_HIDDEN_DIM,
    num_layers=2,
    embedding_dim_other=32,
    # --- End Model architecture parameters ---
    device_str='cuda'
    ):
    """Evaluates a trained FraudGNN model on the test set."""
    print(f"--- Starting GNN Evaluation (Model Type: {model_type}, Conv Type: {conv_type}) ---")
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Processors (for encoder_info and num_numerical_features)
    print(f"Loading processors from: {processors_path}")
    try:
        processors = joblib.load(processors_path)
    except FileNotFoundError:
        print(f"Error: Processors file not found at {processors_path}")
        return
    encoder_gnn_info = processors.get('encoder_gnn')
    num_numerical_features = processors.get('num_numerical_features')
    if encoder_gnn_info is None:
        raise ValueError("GNN encoder info ('encoder_gnn') not found in processors file.")
    if num_numerical_features is None:
        raise ValueError("Number of numerical features ('num_numerical_features') not found in processors file.")
    encoder_gnn_info['num_numerical_features'] = num_numerical_features # Add it for FeatureEncoder

    # 2. Instantiate Dataset (loads/processes graph)
    print("Initializing GNN Dataset...")
    try:
        dataset_root = os.path.dirname(config.PROCESSED_DATA_DIR)
        dataset = IeeeFraudDetectionDataset(
            root=dataset_root,
            processed_data_path=processed_data_path,
            processors_path=processors_path,
        )
        data = dataset[0].to(device) # Get the single HeteroData object
    except FileNotFoundError as e:
         print(f"Error initializing dataset: {e}")
         print("Ensure preprocessing and dataset processing have run successfully.")
         return
    except Exception as e:
         print(f"An unexpected error occurred during dataset initialization: {e}")
         raise

    print("Graph data loaded.")
    # Verify essential components exist for evaluation
    if 'transaction' not in data.node_types:
         print("Error: 'transaction' node type not found in graph data.")
         return
    if not hasattr(data['transaction'], 'test_mask') or not hasattr(data['transaction'], 'y'):
         print("Error: Missing test_mask or y on transaction nodes.")
         return
    if data['transaction'].test_mask.sum() == 0:
         print("Error: No nodes found in the test mask. Cannot evaluate.")
         return
    
    num_nodes_dict = {node_type: data[node_type].num_nodes
                    for node_type in data.node_types if hasattr(data[node_type], 'num_nodes')}
    print(f"Number of nodes per type: {num_nodes_dict}")

    # 3. Instantiate Model (using parameters matching the saved model)
    print("Instantiating model structure...")
    try:
        model = FraudGNN(
            node_metadata=data.metadata(),
            num_nodes_dict=num_nodes_dict,
            encoder_info=encoder_gnn_info,
            hidden_channels=hidden_channels,
            out_channels=2, # Binary classification
            num_layers=num_layers,
            embedding_dim_other=embedding_dim_other,
            model_type=model_type,
            conv_type=conv_type
        ).to(device)
    except Exception as e:
         print(f"Error during model instantiation: {e}")
         raise

    # 4. Load trained weights
    print(f"Loading trained model weights from: {model_path}")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model weights file not found at {model_path}")
        return
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        # This often happens if the architecture doesn't match the saved weights
        print("Ensure the model architecture parameters (hidden_dim, num_layers, etc.) match the trained model.")
        return
    model.eval() # Set model to evaluation mode

    # 5. Perform Inference
    print("Performing inference on the graph...")
    eval_start_time = time.time()
    with torch.no_grad():
        try:
            out = model(data) # Pass the full HeteroData object
        except Exception as e:
             print(f"Error during forward pass for evaluation: {e}")
             return
    eval_duration = time.time() - eval_start_time
    print(f"Inference completed in {eval_duration:.2f}s")

    # 6. Extract Test Predictions and Labels
    test_mask = data['transaction'].test_mask
    target = data['transaction'].y
    test_logits = out[test_mask]
    test_probs = F.softmax(test_logits, dim=1)[:, 1] # Prob of class 1 (Fraud)
    test_pred_class = (test_probs > 0.5).int()
    y_test = target[test_mask]

    # Move results to CPU for scikit-learn metrics
    y_test_np = y_test.cpu().numpy()
    test_probs_np = test_probs.cpu().numpy()
    test_pred_class_np = test_pred_class.cpu().numpy()

    # 7. Calculate Metrics
    print("Calculating evaluation metrics...")
    try:
        # Ensure there are both classes in test set for AUC/AP
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

    # 8. Print Results
    print("\n--- GNN Evaluation Results ---")
    print(f"Model Type: {model_type}, Conv Type: {conv_type}")
    print(f"Test AUC: {auc:.4f}")
    print(f"Test Average Precision (AP): {ap:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    if cm.shape == (2, 2): # Ensure standard 2x2 matrix
        print(f"   TN: {cm[0][0]:<6}  FP: {cm[0][1]:<6}")
        print(f"   FN: {cm[1][0]:<6}  TP: {cm[1][1]:<6}")
    else: # Handle cases where confusion matrix might be smaller (e.g., only one class predicted)
         print(cm)
    print("----------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained FraudGNN model.")
    parser.add_argument("--data_path", type=str, default=config.PROCESSED_DATA_PATH, help="Path to processed data pickle (used by dataset).")
    parser.add_argument("--proc_path", type=str, default=config.PROCESSORS_PATH, help="Path to processors joblib (used by dataset).")
    parser.add_argument("--model_path", type=str, default=config.GNN_MODEL_PATH, help="Path to trained GNN model state dict.")
    # --- Arguments MUST match the trained model's architecture ---
    parser.add_argument("--model_type", type=str, default='hetero', choices=['hetero', 'homo'], help="Type of GNN core used during training.")
    parser.add_argument("--conv_type", type=str, default='SAGE', choices=['RGCN', 'SAGE', 'GAT'], help="Type of GNN convolution layer used during training.")
    parser.add_argument("--hidden_dim", type=int, default=config.GNN_HIDDEN_DIM, help="Hidden dimension size used during training.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers used during training.")
    parser.add_argument("--emb_dim_other", type=int, default=32, help="Embedding dimension for non-transaction nodes used during training.")
    # --- End Architecture Arguments ---
    parser.add_argument("--device", type=str, default='cuda', help="Device to use ('cuda' or 'cpu').")
    args = parser.parse_args()

    evaluate_gnn(
        processed_data_path=args.data_path,
        processors_path=args.proc_path,
        model_path=args.model_path,
        model_type=args.model_type,
        conv_type=args.conv_type,
        hidden_channels=args.hidden_dim,
        num_layers=args.num_layers,
        embedding_dim_other=args.emb_dim_other,
        device_str=args.device
    )
