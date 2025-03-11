
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
import torch_geometric
from torch_geometric.nn import HeteroConv, RGCNConv, GATConv, SAGEConv, Linear # Import desired layers
from torch_geometric.data import HeteroData, Data



class FeatureEncoder(nn.Module):
    """
    Encodes raw node features and creates initial representations.
    - Handles numerical + categorical embeddings for 'transaction' nodes.
    - Handles embeddings for other node types.
    """
    def __init__(self, node_types_list, num_nodes_dict, encoder_info, embedding_dim_other_nodes, hidden_dim):
        super().__init__()
        self.node_types = node_types_list  # List of node types from data.metadata()
        self.num_nodes_dict = num_nodes_dict # Store the dict with node counts

        self.encoder_info = encoder_info # From preprocessing (contains info for transaction categoricals)
        self.embedding_dim_other_nodes = embedding_dim_other_nodes
        self.hidden_dim = hidden_dim # Target dimension after encoding

        self.embeddings_other = nn.ModuleDict()
        self.embeddings_trans_cat = nn.ModuleDict()
        self.linear_maps = nn.ModuleDict()

        total_trans_input_dim = 0

       # Transaction Node Feature Processing 
        trans_num_features = encoder_info.get('num_numerical_features', 0) # Store this during preprocessing!
        trans_cat_cols = list(encoder_info.get('embedding_dims', {}).keys())
        total_trans_input_dim += trans_num_features

        if trans_cat_cols:
            print(f"  FeatureEncoder: Creating transaction categorical embeddings for: {trans_cat_cols}")
            for col in trans_cat_cols:
                cardinality = encoder_info.get('categories', {}).get(col, 0)
                if cardinality == 0: continue
                num_embeddings = cardinality + 1 # +1 for unknown (-1 index)
                embedding_dim = encoder_info.get('embedding_dims', {}).get(col, 0)
                if embedding_dim == 0: continue

                self.embeddings_trans_cat[col] = nn.Embedding(num_embeddings, embedding_dim)
                total_trans_input_dim += embedding_dim
                print(f"    Transaction emb '{col}': {num_embeddings} cats -> {embedding_dim} dims")

        # Linear layer to project combined transaction features to hidden_dim
        if total_trans_input_dim > 0:
             self.linear_maps['transaction'] = Linear(total_trans_input_dim, hidden_dim)
             print(f"  FeatureEncoder: Transaction input dim {total_trans_input_dim} -> {hidden_dim}")
        else:
             # Handle case where transaction nodes might *only* get ID embedding? Unlikely here.
             print("  FeatureEncoder: Warning - No initial features defined for transaction nodes.")


       # Other Node Type Embeddings 
        for node_type in self.node_types: # Iterate directly through the string names
            print(f"  Processing node_type: '{node_type}'") # Keep for debugging if needed

            if node_type == 'transaction': continue

            # Get num_nodes directly from the passed dictionary using the string key
            num_nodes = self.num_nodes_dict.get(node_type, 0)

            if num_nodes == 0:
                 print(f"  FeatureEncoder: Warning - num_nodes is 0 for '{node_type}'. Skipping embedding.")
                 continue

            # Simple embedding based on node ID for other types
            self.embeddings_other[node_type] = nn.Embedding(num_nodes, self.embedding_dim_other_nodes)
            # Linear layer to project to hidden_dim
            self.linear_maps[node_type] = Linear(self.embedding_dim_other_nodes, self.hidden_dim)
            print(f"  FeatureEncoder: Embedding for '{node_type}': {num_nodes} nodes -> {self.embedding_dim_other_nodes} dims -> {self.hidden_dim} dims")


    # def forward(self, x_dict, n_id_dict=None):
    #     """
    #     Generates initial node embeddings/features.

    #     Args:
    #         x_dict (dict): Input dict potentially containing 'transaction' features:
    #                        {'transaction': {'x_num': Tensor, 'x_cat': Tensor}}
    #                        Other node types might be None initially.
    #         n_id_dict (dict, optional): Node IDs for sampled batches. If None, assumes full batch.

    #     Returns:
    #         dict: Output dictionary with node features projected to hidden_dim.
    #               {'node_type': Tensor[num_nodes, hidden_dim]}
    #     """
    #     out_x_dict = {}

    #     try:
    #         device = next(self.parameters()).device
    #     except StopIteration:
    #         # Handle case where model might have NO parameters (unlikely for this setup)
    #         # Try getting device from input tensors if available
    #         if 'transaction' in x_dict and x_dict['transaction'].get('x_num') is not None:
    #             device = x_dict['transaction']['x_num'].device
    #         elif 'transaction' in x_dict and x_dict['transaction'].get('x_cat') is not None:
    #             device = x_dict['transaction']['x_cat'].device
    #         else:
    #             # Fallback to CPU if no parameters or input tensors found
    #             print("Warning: Could not determine device in FeatureEncoder. Defaulting to CPU.")
    #             device = torch.device('cpu')        

    #     # Process Transaction Nodes
    #     if 'transaction' in self.node_types:
    #         trans_features = []
    #         trans_input = x_dict.get('transaction', {})
    #         x_num = trans_input.get('x_num')
    #         x_cat = trans_input.get('x_cat') # Integer encoded categorical features

    #         if x_num is not None:
    #             trans_features.append(x_num)

    #         if x_cat is not None and self.embeddings_trans_cat:
    #             trans_cat_cols = list(self.encoder_info.get('embedding_dims', {}).keys())
    #             if x_cat.shape[1] == len(trans_cat_cols):
    #                 for i, col in enumerate(trans_cat_cols):
    #                     if col in self.embeddings_trans_cat:
    #                         cat_indices = x_cat[:, i] + 1 # Shift index for embedding lookup
    #                         num_embeddings = self.embeddings_trans_cat[col].num_embeddings
    #                         cat_indices = torch.clamp(cat_indices, 0, num_embeddings - 1)
    #                         trans_features.append(self.embeddings_trans_cat[col](cat_indices))
    #             else:
    #                  print("Warning: Mismatch between x_cat columns and expected categorical columns.")

    #         if trans_features:
    #             combined_trans_features = torch.cat(trans_features, dim=1)
    #             if 'transaction' in self.linear_maps:
    #                  out_x_dict['transaction'] = self.linear_maps['transaction'](combined_trans_features)
    #             else: # Should not happen if configured correctly
    #                  out_x_dict['transaction'] = combined_trans_features # Pass through if no linear map
    #         elif 'transaction' in self.linear_maps:
    #              # If transaction nodes have *no* input features but an embedding was expected
    #              # (e.g. based only on ID), handle that here. Less likely for this problem.
    #              print("Warning: No transaction features found, cannot apply linear map.")


    #     # Process Other Node Types (using ID embeddings)
    #     for node_type in self.node_types:
    #         if node_type == 'transaction': continue
    #         if node_type not in self.embeddings_other: continue # Skip if no embedding created

    #         if n_id_dict is not None: # Sampled batch
    #             if node_type in n_id_dict:
    #                 node_ids = n_id_dict[node_type]
    #                 x = self.embeddings_other[node_type](node_ids)
    #             else:
    #                 # Node type not present in the sampled batch
    #                 num_nodes_in_batch = x_dict[node_type].shape[0] # Get expected size from input dict
    #                 x = torch.zeros((num_nodes_in_batch, self.embedding_dim_other_nodes), device=self.embeddings_other[node_type].weight.device) # Placeholder zeros
    #         else: # Full batch
    #             x = self.embeddings_other[node_type].weight # Use all embeddings

    #         # Apply linear projection
    #         if node_type in self.linear_maps:
    #              out_x_dict[node_type] = self.linear_maps[node_type](x)
    #         else: # Pass through if no linear map (shouldn't happen here)
    #              out_x_dict[node_type] = x


    #     return out_x_dict

    def forward(self, x_dict, n_id_dict=None):
        """
        Generates initial node embeddings/features projected to hidden_dim.
        Ensures all node types present in self.node_types have an entry in the output dict.
        """
        out_x_dict = {}

        # This assumes the model has parameters, which it should after init
        try:
            device = next(self.parameters()).device
        except StopIteration:
            # Handle case where model might have NO parameters (unlikely for this setup)
            # Try getting device from input tensors if available
            if 'transaction' in x_dict and x_dict['transaction'].get('x_num') is not None:
                device = x_dict['transaction']['x_num'].device
            elif 'transaction' in x_dict and x_dict['transaction'].get('x_cat') is not None:
                device = x_dict['transaction']['x_cat'].device
            else:
                # Fallback to CPU if no parameters or input tensors found
                print("Warning: Could not determine device in FeatureEncoder. Defaulting to CPU.")
                device = torch.device('cpu')


       # Process All Node Types 
        for node_type in self.node_types:
            # Initialize placeholder for this node type's features
            final_node_features = None

            if node_type == 'transaction':
               # Transaction Node Processing 
                trans_features = []
                # Use .get() safely for potentially missing keys/values
                trans_input = x_dict.get('transaction', {})
                x_num = trans_input.get('x_num')
                x_cat = trans_input.get('x_cat')

                if x_num is not None:
                    trans_features.append(x_num)

                if x_cat is not None and self.embeddings_trans_cat:
                    trans_cat_cols = list(self.encoder_info.get('embedding_dims', {}).keys())
                    # Check if x_cat has columns before accessing shape[1]
                    if x_cat.shape[1] > 0 and x_cat.shape[1] == len(trans_cat_cols):
                        for i, col in enumerate(trans_cat_cols):
                            if col in self.embeddings_trans_cat:
                                cat_indices = x_cat[:, i] + 1
                                num_embeddings = self.embeddings_trans_cat[col].num_embeddings
                                cat_indices = torch.clamp(cat_indices, 0, num_embeddings - 1)
                                trans_features.append(self.embeddings_trans_cat[col](cat_indices))
                    elif x_cat.shape[1] > 0: # Only warn if x_cat is not empty
                         print(f"Warning: Mismatch between x_cat columns ({x_cat.shape[1]}) and expected categorical columns ({len(trans_cat_cols)}) for transaction.")

                if trans_features:
                    combined_trans_features = torch.cat(trans_features, dim=1)
                    if 'transaction' in self.linear_maps:
                         final_node_features = self.linear_maps['transaction'](combined_trans_features)
                    else:
                         final_node_features = combined_trans_features
                # else: # No transaction features found - will be handled below

            else:
               # Other Node Type Processing 
                if node_type in self.embeddings_other: # Check if embedding layer exists
                    x_emb = None
                    if n_id_dict is not None: # Sampled batch
                        if node_type in n_id_dict:
                            node_ids = n_id_dict[node_type]
                            # Ensure node_ids is not empty before embedding lookup
                            if len(node_ids) > 0:
                                 x_emb = self.embeddings_other[node_type](node_ids)
                            else: # Empty list of node IDs for this type in the batch
                                 x_emb = torch.zeros((0, self.embedding_dim_other_nodes), device=device)
                        # else: node_type not in this batch sample - handled below if needed
                    else: # Full batch
                        # Check if the embedding layer actually has weights (num_nodes > 0)
                        if self.embeddings_other[node_type].num_embeddings > 0:
                             x_emb = self.embeddings_other[node_type].weight
                        else: # num_nodes was 0, create empty tensor
                             x_emb = torch.zeros((0, self.embedding_dim_other_nodes), device=device)

                    # Apply linear projection if embedding was generated
                    if x_emb is not None and node_type in self.linear_maps:
                         # Ensure input to linear map is not empty
                         if x_emb.shape[0] > 0:
                              final_node_features = self.linear_maps[node_type](x_emb)
                         else: # Input is empty, output should be empty too
                              final_node_features = torch.zeros((0, self.hidden_dim), device=device)
                    elif x_emb is not None: # Pass through embedding if no linear map
                         final_node_features = x_emb
                # else: # No embedding created for this node type (num_nodes=0) - handled below

           # Ensure Output Tensor Exists for this node_type 
            if final_node_features is None:
                # Determine the expected number of nodes for the output tensor
                current_num_nodes = 0
                if n_id_dict is not None: # Sampled batch
                    # If node_type was in n_id_dict, len(node_ids) was used above.
                    # If not in n_id_dict, assume 0 nodes for this type in this batch.
                    current_num_nodes = len(n_id_dict.get(node_type, []))
                else: # Full batch
                    current_num_nodes = self.num_nodes_dict.get(node_type, 0)

                # Create a zero tensor with the correct shape [num_nodes, hidden_dim]
                # print(f"   FeatureEncoder: Creating zero tensor for node type '{node_type}' (shape: ({current_num_nodes}, {self.hidden_dim}))")
                final_node_features = torch.zeros((current_num_nodes, self.hidden_dim), device=device)

           # Assign to output dictionary 
            out_x_dict[node_type] = final_node_features
           # End Loop 

       # Final Check (Optional Safeguard) 
        # Ensure all node types from metadata are present in the output dict
        for node_type in self.node_types:
             if node_type not in out_x_dict:
                  print(f"   FeatureEncoder: Warning - Node type '{node_type}' still missing from output dict after processing loop. Creating empty tensor.")
                  out_x_dict[node_type] = torch.zeros((0, self.hidden_dim), device=device)

        return out_x_dict


class HeteroGNN(nn.Module):
    """Heterogeneous GNN using HeteroConv."""
    def __init__(self, hetero_metadata, hidden_channels, out_channels, num_layers=2, conv_type='RGCN', heads=4):
        super().__init__()
        self.convs = ModuleList()
        for i in range(num_layers):
            # Input is output of FeatureEncoder or previous layer
            in_dim = hidden_channels 
            # Output dim for GAT with concat=False is just out_channels
            current_out_channels = hidden_channels if i < num_layers - 1 else out_channels

            # Create convolution layer based on type
            if conv_type == 'RGCN':
                num_relations = len(hetero_metadata[1])
                conv = RGCNConv(in_dim, current_out_channels, num_relations=num_relations, num_bases=4) # num_bases can be tuned
            elif conv_type == 'SAGE':
                conv = SAGEConv(in_dim, current_out_channels, aggr='mean')
            elif conv_type == 'GAT':
               # MODIFIED: Use heads, ensure concat=False if output dim should be current_out_channels 
                # If concat=True, output dim would be current_out_channels * heads
                conv = GATConv(in_dim, current_out_channels, heads=heads, concat=False, dropout=0.5) # Using concat=False
               # END MODIFIED 
            else:
                raise ValueError(f"Unsupported conv_type: {conv_type}")

        self.num_layers = num_layers

    def forward(self, x_dict, edge_index_dict):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < self.num_layers - 1:
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=0.5, training=self.training) for key, x in x_dict.items()}
        return x_dict


class HomoGNN(nn.Module):
    """Homogeneous GNN stack (e.g., GAT or SAGE)."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, conv_type='GAT', heads=4):
        super().__init__()
        self.convs = ModuleList()
        print(f"  Initializing HomoGNN with in_channels={in_channels}") # Debug print
        for i in range(num_layers):
            current_in_channels = in_channels if i == 0 else hidden_channels
            current_out_channels = hidden_channels if i < num_layers - 1 else out_channels

            if conv_type == 'GAT':
                self.convs.append(GATConv(current_in_channels, current_out_channels, heads=heads, concat=False, dropout=0.5))
            elif conv_type == 'SAGE':
                 self.convs.append(SAGEConv(current_in_channels, current_out_channels, aggr='mean'))
            else:
                 raise ValueError(f"Unsupported conv_type: {conv_type}")


        self.num_layers = num_layers

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                # Apply dropout after activation                
                x = F.dropout(x, p=0.5, training=self.training)
        return x

class FraudGNN(nn.Module):
    """
    Main GNN model integrating feature encoding and graph convolutions.
    Can switch between Heterogeneous and Homogeneous GNN cores.
    """
    def __init__(self,
                 node_metadata,      # From data.metadata()
                 num_nodes_dict,     # Number of nodes per type (dict)
                 encoder_info,       # From preprocessing (for transaction features)
                 hidden_channels,    # GNN hidden dimension
                 out_channels,       # Final output dimension (e.g., 2 for binary classification)
                 num_layers=2,       # Number of GNN layers
                 embedding_dim_other=32, # Embedding dim for non-transaction nodes
                 model_type='hetero', # 'hetero' or 'homo'
                 conv_type='SAGE',    # Type of convolution ('RGCN', 'SAGE', 'GAT')
                 heads=4                
                ):
        super().__init__()
        self.model_type = model_type.lower()

        # Determine self.node_types from metadata first
        meta_nodes = node_metadata[0]
        if isinstance(meta_nodes, list) and len(meta_nodes) > 0:
            if isinstance(meta_nodes[0], list):
                # Assume structure is [['type1', 'type2', ...]]
                print("Metadata[0] detected as nested list, extracting inner list.")
                self.node_types = meta_nodes[0]
            elif all(isinstance(item, str) for item in meta_nodes):
                # Assume structure is ['type1', 'type2', ...]
                print("Metadata[0] detected as flat list of strings.")
                self.node_types = meta_nodes
            else:
                 raise TypeError(f"First element of metadata[0] has unexpected type: {type(meta_nodes[0])}. Expected list or str.")
        else:
            raise TypeError(f"Unexpected format or empty list for node types in metadata[0]: {meta_nodes}")

        # Ensure self.node_types is now definitely a list of strings
        if not isinstance(self.node_types, list) or not all(isinstance(item, str) for item in self.node_types):
             raise TypeError(f"Failed to extract a flat list of string node types. Got: {self.node_types}")

        self.hidden_channels = hidden_channels

        # 1. Initial Feature Encoder
        # Projects all input features/embeddings to hidden_channels
        self.feature_encoder = FeatureEncoder(
            node_types_list=self.node_types, 
            num_nodes_dict=num_nodes_dict,   
            encoder_info=encoder_info,
            embedding_dim_other_nodes=embedding_dim_other,
            hidden_dim=hidden_channels
        )

        # 2. Core GNN Layers
        if self.model_type == 'hetero':
            # Pass full metadata here as HeteroGNN might need edge type info
            self.gnn = HeteroGNN(node_metadata, hidden_channels, hidden_channels, # Output is hidden_channels
                                 num_layers, conv_type, heads=heads)
            # HeteroGNN outputs hidden_channels
            gnn_out_dim = hidden_channels 
        elif self.model_type == 'homo':
           # HomoGNN now takes hidden_channels as input 
            print(f"  Instantiating HomoGNN with input_channels={hidden_channels}")
            self.gnn = HomoGNN(
                in_channels=hidden_channels, # Input comes from FeatureEncoder
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=num_layers,
                conv_type=conv_type,
                heads=heads
            )
            gnn_out_dim = hidden_channels
        else:
            raise ValueError("model_type must be 'hetero' or 'homo'")

        # 3. Final Classification Head (applied to transaction nodes)
        self.final_layer = Linear(gnn_out_dim, out_channels)




    def forward(self, data, x_dict_encoded=None, x_homo=None, homo_structure = None, n_id_dict=None):
        # Forward pass. Handles both HeteroData and pre-converted Data.
 
        # 1. Encode initial features for ALL node types using FeatureEncoder
        #    Prepare input dict for encoder based on the original data structure
        #    (even if 'data' is already homogeneous, we need original features for encoder)
        x_input_dict = {}
        original_data = data # Keep reference if data gets overwritten
        if isinstance(data, Data) and hasattr(data, '_hetero_data'): # Heuristic to get original if converted
             original_data = data._hetero_data # Access potentially stored original (use with caution)
             print("Warning: Accessing _hetero_data, might be unstable.")
        elif not isinstance(data, HeteroData) and self.model_type == 'hetero':
             raise TypeError("Hetero model requires HeteroData input.")
        # It's better to pass the original HeteroData to forward even in homo mode,
        # let train_gnn handle the conversion *after* getting features if needed.
        # Let's revert to assuming 'data' passed here is ALWAYS the initial HeteroData


        # 1. Apply GNN layers
        x_transaction = None # Initialize

        if self.model_type == 'hetero':

            if x_dict_encoded is None:
                raise ValueError("Hetero mode requires pre-encoded 'x_dict_encoded'.")
            if not isinstance(data, HeteroData):
                 raise TypeError("Hetero model requires original HeteroData as 'data' input.")

            # Apply HeteroGNN layers to pre-encoded features
            x_dict = self.gnn(x_dict_encoded, data.edge_index_dict)
            x_transaction = x_dict.get('transaction')
            if x_transaction is None:
                 raise ValueError("HeteroGNN did not produce output for 'transaction' node type.")


        elif self.model_type == 'homo':

            if x_homo is None:
                raise ValueError("Homo mode requires pre-constructed 'x_homo' tensor.")
            # Use homo_structure if provided, otherwise assume 'data' is the structure
            structure_to_use = homo_structure if homo_structure is not None else data
            if not isinstance(structure_to_use, Data):
                 raise TypeError(f"Homo model requires Data structure input, got {type(structure_to_use)}.")
            if not hasattr(structure_to_use, 'edge_index'):
                 raise ValueError("Homogeneous structure missing 'edge_index'.")
            if not hasattr(structure_to_use, 'node_type'):
                 raise ValueError("Homogeneous structure missing 'node_type'.")

            # Apply HomoGNN layers using pre-constructed x_homo
            x_processed = self.gnn(x_homo, structure_to_use.edge_index)

            # Extract transaction node outputs using node_type mapping
            try:
                trans_node_type_idx = self.node_types.index('transaction')
            except ValueError:
                raise ValueError("'transaction' not found in original node type list stored in model.")

            trans_node_mask = (structure_to_use.node_type == trans_node_type_idx)
            x_transaction = x_processed[trans_node_mask]

            if x_transaction.shape[0] == 0:
                 print("Warning: No transaction node outputs found after HomoGNN processing and masking.")
        else:
             raise ValueError(f"Invalid model_type: {self.model_type}")

        # 4. Apply final layer to transaction node outputs
        # Ensure x_transaction is valid before applying final layer
        if x_transaction is None:
             raise ValueError("x_transaction is None before final layer. Check GNN processing paths.")
        
        if x_transaction.shape[0] > 0: # Only apply if there are transaction outputs
             out = self.final_layer(x_transaction)
        else: # Handle empty input to final layer
             print("Warning: x_transaction input to final_layer is empty. Returning empty tensor.")
             # Get output features from the layer definition
             out_features = self.final_layer.out_features
             # Get device from a parameter if possible, otherwise use input device
             try:
                 device = next(self.final_layer.parameters()).device
             except StopIteration:
                 device = x_transaction.device # Fallback to input device
             out = torch.zeros((0, out_features), device=device) # Create empty output tensor


        return out # Return logits for transaction nodes
