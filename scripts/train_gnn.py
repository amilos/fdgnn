
import torch_geometric; 
print(torch_geometric.__version__)
from pathlib import Path
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch_geometric.data import HeteroData, Data
# from torch_geometric.loader import NeighborLoader # Keep if using sampling later
import pickle
import joblib
import argparse
from sklearn.metrics import roc_auc_score
import os
import sys
import time # For timing epochs

# Add the parent directory to the Python path so 'src' can be found
sys.path.append(str(Path(__file__).parent.parent))

import src.config as config

from src.gnn_dataset import IeeeFraudDetectionDataset

# Import your specific GNN model class(es)
from src.gnn_model import FraudGNN # Import the main wrapper model

def train_gnn(
    processed_data_path,
    processors_path,
    model_output_path,
    model_type='hetero', # 'hetero' or 'homo'
    conv_type='SAGE',    # 'RGCN', 'SAGE', 'GAT' (used by FraudGNN)
    hidden_channels=config.GNN_HIDDEN_DIM,
    num_layers=2,
    embedding_dim_other=32, # Embedding dim for non-transaction nodes
    gat_heads=4, # Keep GAT heads param if using GAT    
    learning_rate=config.GNN_LEARNING_RATE,
    epochs=config.GNN_EPOCHS,
    patience=config.GNN_PATIENCE,
    use_scheduler=True, # Flag to enable/disable scheduler
    scheduler_factor=config.GNN_SCHEDULER_FACTOR,
    scheduler_patience=config.GNN_SCHEDULER_PATIENCE,
    scheduler_min_lr=config.GNN_SCHEDULER_MIN_LR,    
    device_str='mps'
    ):
 

    print(f"--- Starting GNN Training (Model Type: {model_type}, Conv Type: {conv_type}) ---")
    # Set device with support for CUDA, MPS (Apple Silicon), or CPU fallback
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    # elif device_str == 'mps' and hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    #     device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # 1. Load Processors (for encoder_info)
    print(f"Loading processors from: {processors_path}")
    try:
        processors = joblib.load(processors_path)
    except FileNotFoundError:
        print(f"Error: Processors file not found at {processors_path}")
        return
    encoder_gnn_info = processors.get('encoder_gnn')
    if encoder_gnn_info is None:
        raise ValueError("GNN encoder info ('encoder_gnn') not found in processors file.")
    # --- Get number of numerical features (MUST be stored during preprocessing) ---
    # Add this to your preprocess_main.py when saving processors:
    # processors['num_numerical_features'] = len(current_num_cols) # Or however you track it
    num_numerical_features = processors.get('num_numerical_features')
    if num_numerical_features is None:
         raise ValueError("Number of numerical features ('num_numerical_features') not found in processors file. Please save it during preprocessing.")
    encoder_gnn_info['num_numerical_features'] = num_numerical_features # Add it for FeatureEncoder


    # 2. Instantiate Dataset (loads/processes graph)
    print("Initializing GNN Dataset...")
    try:
        # Pass the directory containing the 'processed' folder as root
        dataset_root = os.path.dirname(config.PROCESSED_DATA_DIR)
        dataset = IeeeFraudDetectionDataset(
            root=dataset_root,
            processed_data_path=processed_data_path,
            processors_path=processors_path,
            # graph_save_path defaults to config.GRAPH_DATA_PATH
        )
        hetero_data = dataset[0].to(device) # Get the single HeteroData object
    except FileNotFoundError as e:
         print(f"Error initializing dataset: {e}")
         print("Ensure preprocessing and dataset processing have run successfully.")
         return
    except Exception as e:
         print(f"An unexpected error occurred during dataset initialization: {e}")
         raise # Re-raise other errors

    print("Graph data loaded:")
    print("Initial data object type:", type(hetero_data))    
    print(hetero_data)
    print("Metadata loaded:", hetero_data.metadata())

    # Store original metadata for later use
    if isinstance(hetero_data, HeteroData):
        original_metadata = hetero_data.metadata()
        print("Original Metadata stored:", original_metadata)
    else:
        raise TypeError("Initial data loaded is not HeteroData!")



    # Verify essential components exist
    if 'transaction' not in hetero_data.node_types:
         print("Error: 'transaction' node type not found in graph data.")
         return
    if not hasattr(hetero_data['transaction'], 'train_mask') or \
       not hasattr(hetero_data['transaction'], 'val_mask') or \
       not hasattr(hetero_data['transaction'], 'y'):
         print("Error: Missing train_mask, val_mask, or y on transaction nodes.")
         return


    # Add number of nodes per type
    num_nodes_dict = {node_type: hetero_data[node_type].num_nodes
                      for node_type in hetero_data.node_types if hasattr(hetero_data[node_type], 'num_nodes')}
    print(f"Number of nodes per type: {num_nodes_dict}")

    data = hetero_data # Keep original HeteroData

    # Convert to homogeneous if needed
    is_hetero = True
    homo_data_structure = None # Initialize
    if model_type == 'homo':
        homo_data_structure = convert_to_homogeneous(hetero_data)
        if not isinstance(homo_data_structure, Data):
             raise TypeError(f"Conversion to homogeneous failed, type is {type(homo_data_structure)}")
        is_hetero = False

    # 3. Instantiate Model
    print("Instantiating model...")
    try:
        model = FraudGNN(
            node_metadata=original_metadata, # Always pass original metadata
            num_nodes_dict=num_nodes_dict,
            encoder_info=encoder_gnn_info,
            hidden_channels=hidden_channels,
            out_channels=2,
            num_layers=num_layers,
            embedding_dim_other=embedding_dim_other,
            model_type=model_type, # Tell model its type
            conv_type=conv_type,
            heads=gat_heads,
            # No homo_input_channels needed here anymore
        ).to(device)
    except Exception as e:
         print(f"Error during model instantiation: {e}")
         # Print relevant info for debugging
         print("Encoder Info:", encoder_gnn_info)
         raise

    print("Model instantiated:")
    # print(model) # Can be very verbose


    # Pre-calculate Features
    print("Pre-calculating initial node features using FeatureEncoder...")
    with torch.no_grad(): # No need for gradients here
        # Prepare input dict for encoder from original hetero_data
        x_input_dict = {}
        if 'transaction' in model.node_types: # Use model.node_types
             x_input_dict['transaction'] = {
                 'x_num': data['transaction'].x_num if hasattr(data['transaction'], 'x_num') else None,
                 'x_cat': data['transaction'].x_cat if hasattr(data['transaction'], 'x_cat') else None
             }
        for node_type in model.node_types:
             if node_type != 'transaction' and node_type not in x_input_dict:
                  x_input_dict[node_type] = data[node_type].x if hasattr(data[node_type], 'x') else None
        # Move input features to the correct device before encoding
        for node_type, features in x_input_dict.items():
             if isinstance(features, dict): # Handle transaction features dict
                  for key, tensor in features.items():
                       if tensor is not None: features[key] = tensor.to(device)
             elif features is not None: # Handle other node features tensor
                  x_input_dict[node_type] = features.to(device)

        # Run encoder once
        x_dict_encoded = model.feature_encoder(x_input_dict) # Get dict of encoded features

    # Pre-calculate x_homo if needed
    x_homo = None
    if model_type == 'homo':
        print("Constructing x_homo tensor from encoded features...")
        num_total_nodes = homo_data_structure.node_type.size(0)
        x_homo = torch.zeros((num_total_nodes, hidden_channels), device=device)
        node_type_errors = False
        for i, node_type in enumerate(model.node_types):
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
  



    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = None
    if use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max', # Reduce LR when metric stops increasing (AUC)
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=scheduler_min_lr,
            verbose=True # Print message when LR is reduced
        )
        print(f"   Using ReduceLROnPlateau scheduler (factor={scheduler_factor}, patience={scheduler_patience})")


    # Consider class weights for imbalanced data based on training label distribution

    print("Calculating class weights...")
    # Class weight calculation usea hetero_data to easily access transaction labels/mask
    if hasattr(hetero_data['transaction'], 'train_mask') and hetero_data['transaction'].train_mask.sum() > 0:
        train_labels = hetero_data['transaction'].y[hetero_data['transaction'].train_mask]
    else:
        print("Warning: No training nodes found for 'transaction' in HeteroData. Cannot calculate weights.")
        train_labels = torch.tensor([], dtype=torch.long, device=device) # Empty tensor


    # TODO: Review
    n_samples = len(train_labels)
    n_class0 = (train_labels == 0).sum().item()
    n_class1 = (train_labels == 1).sum().item()
    if n_class0 > 0 and n_class1 > 0:
        # ... (calculate class_weight) ...
        class_weight = torch.tensor([n_samples / (2 * n_class0), n_samples / (2 * n_class1)], device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    else:
        # ... (use unweighted loss) ...
        criterion = torch.nn.CrossEntropyLoss()


    best_val_auc = 0.0
    epochs_no_improve = 0

    print("Starting training loop...")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        optimizer.zero_grad()

        # In forward pass the FraudGNN model handles both hetero/homo internally based on its init
        # We always pass the HeteroData object 'data'
        
 
        # Forward pass
        # Pass pre-calculated features/structure
        try:
            # Modify model forward signature if needed, or pass via kwargs
            # Let's assume model.forward is adapted to take precomputed features
            if model_type == 'hetero':
                 out = model(data, x_dict_encoded=x_dict_encoded) # Pass encoded dict
            else: # homo
                 out = model(homo_data_structure, x_homo=x_homo) # Pass structure and features

        except Exception as e:
             print(f"\nError during forward pass in epoch {epoch}: {e}")
             raise


        # Calculate loss
        # Loss is always calculated on transaction nodes using original hetero masks/labels
        train_mask = hetero_data['transaction'].train_mask
        target = hetero_data['transaction'].y
        if train_mask.sum() == 0:
             print(f"Warning: No nodes in training mask for epoch {epoch}!")
             loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
             # 'out' from model is already transaction logits
             loss = criterion(out[train_mask], target[train_mask])


        loss.backward()
        optimizer.step()
        train_loss = loss.item()

        # --- Validation ---
        model.eval()
        val_auc = 0.0
        val_loss = 0.0
        val_mask = hetero_data['transaction'].val_mask # Use original val_mask

        if val_mask.sum() > 0: # Only evaluate if validation nodes exist
            with torch.no_grad():
                try:
                    # Pass both data objects for validation forward pass too
                    # Pass pre-calculated features/structure for validation pass
                    if model_type == 'hetero':
                         out_val = model(data, x_dict_encoded=x_dict_encoded)
                    else: # homo
                         out_val = model(homo_data_structure, x_homo=x_homo)

                    target = hetero_data['transaction'].y # Use original target
                    val_loss = criterion(out_val[val_mask], target[val_mask]).item()
                    val_probs = F.softmax(out_val[val_mask], dim=1)[:, 1]
                    val_target = target[val_mask]

                    if len(torch.unique(val_target)) > 1:
                         val_auc = roc_auc_score(val_target.cpu().numpy(), val_probs.cpu().numpy())
                    else:
                         # print("Warning: Only one class present in validation set, AUC is 0.")
                         val_auc = 0.0 # Assign 0 if only one class present
                except Exception as e:
                    print(f"\nError during validation in epoch {epoch}: {e}")
                    # Decide how to handle - skip validation for this epoch?
                    val_auc = 0.0 # Reset AUC on error
                    val_loss = float('inf') # Indicate validation error
        else:
             print("Warning: No nodes in validation mask for epoch {epoch}!")

        epoch_duration = time.time() - epoch_start_time

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | LR: {current_lr:.1e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Time: {epoch_duration:.2f}s")

        # Step the scheduler based on validation AUC
        if use_scheduler and scheduler is not None:
            scheduler.step(val_auc)

        # --- Early Stopping & Model Saving ---
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            print(f"   New best Val AUC! Saving model to {model_output_path}")
            try:
                os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
                torch.save(model.state_dict(), model_output_path)
            except Exception as e:
                 print(f"   Error saving model: {e}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"   Early stopping triggered after {patience} epochs with no improvement.")
            break

        if use_scheduler and current_lr < scheduler_min_lr * 1.01: # Add a small buffer
            print(f"   Learning rate ({current_lr:.1e}) below minimum threshold. Stopping training.")
            break

    print(f"--- GNN Training Finished --- Best Val AUC: {best_val_auc:.4f}")

def convert_to_homogeneous(data: HeteroData): # Remove unused args
    """Converts HeteroData structure to homogeneous Data object."""
    print(f"\n--- Converting HeteroData to Homogeneous (Structure Only) ---")
    if not isinstance(data, HeteroData):
        print("Data is already homogeneous or not HeteroData.")
        return data
    try:
        # --- Perform conversion WITHOUT node_attrs ---
        print("   Converting graph structure using data.to_homogeneous()...")
        homo_data = data.to_homogeneous(add_node_type=True, add_edge_type=True)

        print("Conversion complete. Homogeneous data object:")
        print(homo_data)
        # Verify attributes needed later exist
        if not hasattr(homo_data, 'edge_index'): raise ValueError("edge_index missing")
        if not hasattr(homo_data, 'node_type'): raise ValueError("node_type missing")

        return homo_data
    except Exception as e:
        print(f"Error during data.to_homogeneous() conversion: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FraudGNN model.")
    parser.add_argument("--data_path", type=str, default=config.PROCESSED_DATA_PATH)
    parser.add_argument("--proc_path", type=str, default=config.PROCESSORS_PATH)
    parser.add_argument("--model_path", type=str, default=config.GNN_MODEL_PATH)
    parser.add_argument("--model_type", type=str, default='homo', choices=['hetero', 'homo'])
    parser.add_argument("--conv_type", type=str, default='SAGE', choices=['RGCN', 'SAGE', 'GAT'])
    parser.add_argument("--hidden_dim", type=int, default=config.GNN_HIDDEN_DIM)
    parser.add_argument("--num_layers", type=int, default=config.GNN_NUM_LAYERS)
    parser.add_argument("--emb_dim_other", type=int, default=config.GNN_EMB_DIM_OTHER)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=config.GNN_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=config.GNN_EPOCHS)
    parser.add_argument("--patience", type=int, default=config.GNN_PATIENCE)
    parser.add_argument("--no_scheduler", action='store_true')
    parser.add_argument("--scheduler_factor", type=float, default=config.GNN_SCHEDULER_FACTOR)
    parser.add_argument("--scheduler_patience", type=int, default=config.GNN_SCHEDULER_PATIENCE)
    parser.add_argument("--scheduler_min_lr", type=float, default=config.GNN_SCHEDULER_MIN_LR)
    parser.add_argument("--device", type=str, default='mps')
    args = parser.parse_args()

    train_gnn(
        processed_data_path=args.data_path,
        processors_path=args.proc_path,
        model_output_path=args.model_path,
        model_type=args.model_type,
        conv_type=args.conv_type,
        hidden_channels=args.hidden_dim,
        num_layers=args.num_layers,
        embedding_dim_other=args.emb_dim_other,
        gat_heads=args.gat_heads, # Pass heads
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        use_scheduler=(not args.no_scheduler),
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        scheduler_min_lr=args.scheduler_min_lr,
        device_str=args.device
    )