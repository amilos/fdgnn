# gnn_dataset.py
from pathlib import Path
import torch
from torch_geometric.data import HeteroData, InMemoryDataset
import pandas as pd
import os
import sys
import pickle
import joblib
from tqdm import tqdm # For progress bars

# Add the parent directory to the Python path so 'src' can be found
sys.path.append(str(Path(__file__).parent.parent))

import src.config as config


class IeeeFraudDetectionDataset(InMemoryDataset):
    """
    PyG Dataset class that loads preprocessed data and builds the graph.
    Assumes preprocessing via preprocess_main.py has been run.
    """
    def __init__(
        self,
        root, # Root directory (contains processed folder)
        processed_data_path=config.PROCESSED_DATA_PATH,
        processors_path=config.PROCESSORS_PATH,
        graph_save_path=config.GRAPH_DATA_PATH, # Where to save the final graph
        force_process=True, # Set to True to re-process the graph
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.processed_data_path = processed_data_path
        self.processors_path = processors_path
        self.graph_save_path = graph_save_path # Use specific path for graph

        # Load necessary data/processors immediately for graph building info
        print("Loading preprocessed data and processors for dataset setup...")
        try:
            with open(self.processed_data_path, 'rb') as f:
                self.processed_data = pickle.load(f)
            self.processors = joblib.load(self.processors_path)
        except FileNotFoundError as e:
            print(f"Error: Ensure preprocessing has run. Missing file: {e.filename}")
            print(f"Attempted to load: {self.processed_data_path} and {self.processors_path}")
            raise FileNotFoundError(f"Ensure preprocessing has run. Missing file: {e.filename}") from e

        self.non_target_node_types = config.DEFAULT_NON_TARGET_NODE_TYPES
        self.gnn_exclude_feature_cols = config.GNN_EXCLUDED_COLUMNS
        self.target_col = config.TARGET_COL # Needed for y

        # Adjust root if necessary, InMemoryDataset expects root/processed
        # The 'processed' dir for InMemoryDataset is where the final .pt file goes
        processed_dir_for_pyg = os.path.dirname(self.graph_save_path)
        # The root dir should be the parent of the 'processed' dir
        root_dir_for_pyg = os.path.dirname(processed_dir_for_pyg)

        super().__init__(root_dir_for_pyg, transform, pre_transform, pre_filter) # Use adjusted root

        # Check if processing is needed
        # Note: self.processed_paths is defined by InMemoryDataset based on root and processed_file_names
        needs_processing = force_process or not os.path.exists(self.processed_paths[0])

        if needs_processing:
             print(f"Processing graph data as '{self.processed_paths[0]}' not found or force_process=True.")
             self.process() # Call process if file doesn't exist or forced
        else:
             print(f"Processed graph file found at '{self.processed_paths[0]}'. Loading...")

        try:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
            print("Graph data loaded successfully.")
        except FileNotFoundError:
             print(f"Error: Processed graph file not found at {self.processed_paths[0]} even after checking.")
             print("Please ensure the 'process' method runs correctly and saves the file.")
             raise
        except pickle.UnpicklingError as e: # Catch the specific error if needed
             print(f"Error unpickling graph data from {self.processed_paths[0]}: {e}")
             print("This might happen if the file is corrupted or saved with incompatible versions.")
             print("Try deleting the file and re-running preprocessing/dataset creation.")
             raise
        except Exception as e: # Catch other potential loading errors
             print(f"An unexpected error occurred loading graph data: {e}")
             raise


    @property
    def raw_file_names(self):
        # Points to files within the 'raw' directory relative to self.root
        # Not strictly needed if download isn't implemented, but good practice
        raw_dir = os.path.join(self.root, 'raw') # self.root is root_dir_for_pyg
        return [
            os.path.relpath(config.TRANSACTION_FILE_TRAIN, raw_dir),
            os.path.relpath(config.IDENTITY_FILE_TRAIN, raw_dir)
        ]


    @property
    def processed_file_names(self):
        # This should be the base filename of the final saved graph object
        # It will be saved inside self.processed_dir (root/processed)
        return [os.path.basename(self.graph_save_path)]

    def download(self):
        # Data should be downloaded manually or via another script
        print(f"Raw data download not implemented. Please place files in '{self.raw_dir}'")

    def _extract_features_and_labels(self, split='train'):
        """Helper to get features and labels for a specific split."""
        if split not in self.processed_data:
            raise ValueError(f"Split '{split}' not found in processed_data.")

        df_features_gnn = self.processed_data[split]['gnn'].copy() # Work on a copy
        y_labels = self.processed_data[split]['y']
        split_indices = self.processed_data[split]['index'] # Get original indices

        # Exclude GNN-specific feature columns AFTER scaling/encoding
        cols_to_drop = [col for col in self.gnn_exclude_feature_cols if col in df_features_gnn.columns]
        if cols_to_drop:
            df_features_gnn = df_features_gnn.drop(columns=cols_to_drop)
            # print(f"   GNN Features: Dropped {cols_to_drop} for split '{split}'")

        # Separate numerical and categorical based on the GNN encoder info
        # This is needed if my GNN model expects separate inputs
        encoder_gnn = self.processors.get('encoder_gnn', {})
        cat_cols_encoded = list(encoder_gnn.get('embedding_dims', {}).keys())
        cat_cols_present = [col for col in cat_cols_encoded if col in df_features_gnn.columns]

        num_cols_scaled = [col for col in df_features_gnn.columns if col not in cat_cols_present]

        # Convert to tensors
        x_num_tensor = torch.tensor(df_features_gnn[num_cols_scaled].values, dtype=torch.float32) if num_cols_scaled else None
        # Categorical features should be Long type for embedding lookup
        x_cat_tensor = torch.tensor(df_features_gnn[cat_cols_present].values, dtype=torch.long) if cat_cols_present else None
        y_tensor = torch.tensor(y_labels.values, dtype=torch.long)

        return x_num_tensor, x_cat_tensor, y_tensor, split_indices

    def process(self):
        """Builds the HeteroData graph object from preprocessed data."""
        print("Building HeteroData graph...")
        data = HeteroData()

        # 1. Combine features and labels from train/val/test splits
        print("  Extracting features and labels for all splits...")
        x_num_train, x_cat_train, y_train, train_idx = self._extract_features_and_labels('train')
        x_num_val, x_cat_val, y_val, val_idx = self._extract_features_and_labels('val')
        x_num_test, x_cat_test, y_test, test_idx = self._extract_features_and_labels('test')

        # Combine all indices and create mapping
        all_original_indices = pd.concat([pd.Series(train_idx), pd.Series(val_idx), pd.Series(test_idx)]).unique()
        all_original_indices.sort() # Ensure consistent order
        original_idx_to_new_idx = {orig_idx: new_idx for new_idx, orig_idx in enumerate(all_original_indices)}
        num_total_transaction_nodes = len(all_original_indices)
        print(f"  Total unique transaction nodes: {num_total_transaction_nodes}")

        # Combine features (handle numerical and categorical separately if model needs it)
        # Determine dimensions
        num_numerical_features = x_num_train.shape[1] if x_num_train is not None else 0
        num_categorical_features = x_cat_train.shape[1] if x_cat_train is not None else 0

        # Initialize combined tensors
        combined_x_num = torch.zeros((num_total_transaction_nodes, num_numerical_features), dtype=torch.float32) if num_numerical_features > 0 else None
        combined_x_cat = torch.zeros((num_total_transaction_nodes, num_categorical_features), dtype=torch.long) if num_categorical_features > 0 else None
        combined_y = torch.zeros(num_total_transaction_nodes, dtype=torch.long)

        # Map features and labels to new contiguous indices
        print("  Mapping features and labels to contiguous indices...")
        if combined_x_num is not None:
            combined_x_num[ [original_idx_to_new_idx[i] for i in train_idx] ] = x_num_train
            combined_x_num[ [original_idx_to_new_idx[i] for i in val_idx] ] = x_num_val
            combined_x_num[ [original_idx_to_new_idx[i] for i in test_idx] ] = x_num_test
            data['transaction'].x_num = combined_x_num # Store numerical features

        if combined_x_cat is not None:
            combined_x_cat[ [original_idx_to_new_idx[i] for i in train_idx] ] = x_cat_train
            combined_x_cat[ [original_idx_to_new_idx[i] for i in val_idx] ] = x_cat_val
            combined_x_cat[ [original_idx_to_new_idx[i] for i in test_idx] ] = x_cat_test
            data['transaction'].x_cat = combined_x_cat # Store categorical features (indices)

        combined_y[ [original_idx_to_new_idx[i] for i in train_idx] ] = y_train
        combined_y[ [original_idx_to_new_idx[i] for i in val_idx] ] = y_val
        combined_y[ [original_idx_to_new_idx[i] for i in test_idx] ] = y_test
        data['transaction'].y = combined_y
        data['transaction'].num_nodes = num_total_transaction_nodes

        # 2. Create masks
        print("  Creating train/val/test masks...")
        train_mask = torch.zeros(num_total_transaction_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_total_transaction_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_total_transaction_nodes, dtype=torch.bool)

        train_mask[[original_idx_to_new_idx[i] for i in train_idx]] = True
        val_mask[[original_idx_to_new_idx[i] for i in val_idx]] = True
        test_mask[[original_idx_to_new_idx[i] for i in test_idx]] = True

        data['transaction'].train_mask = train_mask
        data['transaction'].val_mask = val_mask
        data['transaction'].test_mask = test_mask

        # 3. Create Edges
        print("  Building graph edges...")
        full_df_original_ids = self.processed_data['full_df_original_ids']
        # Ensure the snapshot only contains rows corresponding to the nodes we are keeping
        df_for_edges = full_df_original_ids.loc[all_original_indices].copy()
        # Map transaction index to new contiguous index for edge creation
        df_for_edges['new_transaction_idx'] = df_for_edges.index.map(original_idx_to_new_idx)

        for node_type in tqdm(self.non_target_node_types, desc="Building Edges"):
            if node_type not in df_for_edges.columns:
                 print(f"   Warning: Node type column '{node_type}' not found in snapshot. Skipping edges.")
                 continue

            # Map entity IDs to contiguous indices for this node type
            unique_entries = df_for_edges[[node_type]].dropna().drop_duplicates()
            unique_entries = unique_entries.sort_values(by=node_type) # Consistent mapping
            unique_entries['entity_node_idx'] = range(len(unique_entries))
            num_entity_nodes = len(unique_entries)

            if num_entity_nodes == 0:
                 print(f"   No unique, non-null entries found for node type '{node_type}'. Skipping.")
                 continue

            data[node_type].num_nodes = num_entity_nodes
            # Initialize features for entity nodes (e.g., zeros or learnable embeddings later)
            # I might want different feature dimensions per node type
            data[node_type].x = torch.zeros((num_entity_nodes, 1), dtype=torch.float32) # Placeholder feature

            # Merge to get edge pairs (transaction_idx -> entity_idx)
            edge_df = pd.merge(
                df_for_edges[['new_transaction_idx', node_type]].dropna(),
                unique_entries, on=node_type, how='inner'
            )

            if not edge_df.empty:
                # Create edge index tensor [2, num_edges]
                edge_list = torch.tensor([
                    edge_df['new_transaction_idx'].values, # Source: transaction node (contiguous index)
                    edge_df['entity_node_idx'].values     # Target: entity node (contiguous index)
                ], dtype=torch.long)

                # Store edges in HeteroData (Transaction -> Entity)
                edge_type_tuple = ('transaction', f'linked_to_{node_type}', node_type)
                data[edge_type_tuple].edge_index = edge_list
                print(f"   Created {edge_list.shape[1]} edges for {edge_type_tuple}")

                #  Add reverse edges (Entity -> Transaction) if needed by model/layers
                reverse_edge_type_tuple = (node_type, f'linked_from_{node_type}', 'transaction')
                data[reverse_edge_type_tuple].edge_index = edge_list[[1, 0]] # Swap rows
                print(f"   Created {edge_list.shape[1]} reverse edges for {reverse_edge_type_tuple}")

            else:
                 print(f"   No edges created for node type '{node_type}'")


       # End Graph Building Logic 

        print("Validating HeteroData graph structure...")
        try:
            data.validate()
            print("Graph validation successful.")
        except Exception as e:
            print(f"Graph validation failed: {e}")
            # Potentially raise error or handle differently

        if self.pre_filter is not None:
            print("Applying pre_filter...")
            data = self.pre_filter(data)
        if self.pre_transform is not None:
            print("Applying pre_transform...")
            data = self.pre_transform(data)

        # Ensure processed directory exists before saving
        os.makedirs(self.processed_dir, exist_ok=True)
        print(f"Saving processed graph data to {self.processed_paths[0]}")

        print("Metadata within process() before saving:", data.metadata()) 

        # Use collate to handle HeteroData batching if needed, otherwise save directly
        torch.save(self.collate([data]), self.processed_paths[0])
        print("Graph data saved.")

