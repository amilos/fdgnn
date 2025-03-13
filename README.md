# Fraud Detection with Graph Neural Networks
This repository contains a final year project focused on fraud detection in financial transactions using Graph Neural Networks (GNNs). By representing transactions as a graph where users, devices, and transactions form nodes and edges, GNNs can capture complex relationships and patterns that traditional machine learning approaches might miss. The project compares the performance of GNN-based fraud detection against traditional gradient boosting models (XGBoost) to demonstrate the advantages of graph-based approaches for identifying fraudulent activities.

## Project Structure
```
fdgnn/
├── data/                               # Data directory
│   ├── raw/                            # Place raw train/test csv files here
│   └── processed/                      # Output of preprocessing
├── models/                             # Saved trained models
├── notebooks/                          # Jupyter notebooks
│   ├── eda.ipynb                       # EDA notebook
│   ├── model_evaluation.ipynb          # Model evaluation notebook
│   ├── concept_drift_analysis.ipynb    # Concept drift analysis notebook
├── scripts/                            # Scripts for various tasks
│   ├── evaluate_gnn.py                 # GNN evaluation script
│   ├── evaluate_xgb.py                 # XGBoost evaluation script
│   ├── infer_gnn.py                    # GNN inference script
│   ├── infer_xgb.py                    # XGBoost inference script
│   ├── preprocess_main.py              # Runs common preprocessing
│   ├── preprocess_utils.py             # Preprocessing functions
│   ├── train_gnn.py                    # GNN training script
│   └── train_xgb.py                    # XGBoost training script
├── src/                                # Source code for GNN and configurations
│   ├── config.py                       # Constants & configurations
│   ├── gnn_model.py                    # GNN architecture definition
│   ├── gnn_dataset.py                  # PyG Dataset class definition
├── tests/                              # Unit tests for preprocessing utilities
├── app.py                              # Demo app for fraud scoring
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Setup

1.  **Clone/Download:** Get the project files.
2.  **Place Data:** Download the raw data files (`train_transaction.csv`, `train_identity.csv`, etc.) and place them in the `data/raw/` directory.
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Review Configuration:** Check `src/config.py` for default paths, column names, and parameters. Adjust if necessary.

## Workflow

1.  **Run Preprocessing:** Execute the main preprocessing script. This will load raw data, perform common steps (imputation, encoding, scaling), split the data, and save the processed data splits and fitted processors.
    ```bash
    python scripts/preprocess_main.py [--nrows N] # Use --nrows for testing with fewer rows
    ```
    This creates `data/processed/processed_data.pkl` and `data/processed/processors.joblib`.

2.  **Train Models:**
    *   **XGBoost:**
        ```bash
        python scripts/train_xgb.py
        ```
        This loads `data/processed/processed_data.pkl`, trains the model, and saves it to `models/xgboost_model.json`.
    *   **GNN:**
        ```bash
        python scripts/train_gnn.py [--device cpu] # Specify device if needed
        ```
        This loads `data/processed/processed_data.pkl` and `data/processed/processors.joblib`, builds/loads the graph using `src/gnn_dataset.py`, trains the model defined in `src/gnn_model.py`, and saves the best model state to `models/gnn_model.pt`.

3.  **Evaluate Models:**
    *   **XGBoost:**
        ```bash
        python scripts/evaluate_xgb.py
        ```
        Loads the test split from `data/processed/processed_data.pkl` and the saved model to calculate metrics.
    *   **GNN:**
        ```bash
        python scripts/evaluate_gnn.py [--device cpu]
        ```
        Loads the graph data, the saved model, and calculates metrics on the test mask.

4.  **Run Inference:** (Using example test files)
    *   **XGBoost:**
        ```bash
        python scripts/infer_xgb.py
        ```
        Loads the test split and the saved model to generate predictions.
    *   **GNN:**
        ```bash
        python scripts/infer_gnn.py [--device cpu]
        ```
        Loads the graph data and the saved model to generate predictions.
5.  **Run Demo App:**
    ```bash
    streamlit run app.py
    ```
