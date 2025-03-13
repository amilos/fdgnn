import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix # For dynamic thresholding

#  Add project root to path 
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

#  Import config 
try:
    import src.config as config
except ImportError as e:
    st.error(f"Error importing project modules: {e}. Ensure paths are correct.")
    # Define essential variables manually if config fails
    config = type('obj', (object,), {
        'RAW_DATA_DIR': os.path.join(project_root, "data", "raw"),
        'PREDICTION_DIR': os.path.join(project_root, "predictions"),
        'INFERENCE_SAMPLE_TRANS_PATH': os.path.join(project_root, "data", "raw", "test_transaction.csv"),
        'INFERENCE_SAMPLE_ID_PATH': os.path.join(project_root, "data", "raw", "test_identity.csv"),
        'ID_COL': 'TransactionID',
        'TARGET_COL': 'isFraud',
        'TIMESTAMP_COL': 'TransactionDT',
        'DISPLAY_COLS': ['TransactionID', 'TransactionDT', 'isFraud', 'TransactionAmt',
                         'ProductCD', 'card4', 'addr1', 'DeviceType']
    })()
    print("Using manually defined config variables.")

#  Caching Function for Data Loading 
@st.cache_data # Cache the combined data loading
def load_demo_data_with_predictions():
    try:
        # Define paths using config
        trans_path = config.INFERENCE_SAMPLE_TRANS_PATH
        id_path = config.INFERENCE_SAMPLE_ID_PATH
        xgb_pred_path = os.path.join(config.PREDICTION_DIR, "xgb_predictions.csv")
        gnn_pred_path = os.path.join(config.PREDICTION_DIR, "gnn_predictions.csv") # Assuming this name

        print(f"Loading sample transaction data from: {trans_path}")
        df_trans = pd.read_csv(trans_path)
        print(f"Loading sample identity data from: {id_path}")
        df_id = pd.read_csv(id_path)
        # Ensure ID columns are compatible for merging (e.g., both int or both str)
        df_trans[config.ID_COL] = df_trans[config.ID_COL].astype(str)
        df_id[config.ID_COL] = df_id[config.ID_COL].astype(str)
        df_sample = pd.merge(df_trans, df_id, on=config.ID_COL, how='left')
        print(f"Sample data shape after merge: {df_sample.shape}")

        print(f"Loading XGBoost predictions from: {xgb_pred_path}")
        df_xgb_preds = pd.read_csv(xgb_pred_path)
        df_xgb_preds[config.ID_COL] = df_xgb_preds[config.ID_COL].astype(str) # Ensure consistent ID type
        if config.TARGET_COL in df_xgb_preds.columns and 'xgb_score' not in df_xgb_preds.columns:
             df_xgb_preds = df_xgb_preds.rename(columns={config.TARGET_COL: 'xgb_score'})

        print(f"Loading GNN predictions from: {gnn_pred_path}")
        df_gnn_preds = pd.read_csv(gnn_pred_path)
        df_gnn_preds[config.ID_COL] = df_gnn_preds[config.ID_COL].astype(str) # Ensure consistent ID type
        if config.TARGET_COL in df_gnn_preds.columns and 'gnn_score' not in df_gnn_preds.columns:
             df_gnn_preds = df_gnn_preds.rename(columns={config.TARGET_COL: 'gnn_score'})

        print("Merging predictions with sample data...")
        df_demo = pd.merge(df_sample, df_xgb_preds[[config.ID_COL, 'xgb_score']], on=config.ID_COL, how='left')
        df_demo = pd.merge(df_demo, df_gnn_preds[[config.ID_COL, 'gnn_score']], on=config.ID_COL, how='left')
        print(f"Final demo data shape: {df_demo.shape}")

       # Ensure scores are numeric, fill missing XGB with NaN for calculation 
        df_demo['xgb_score'] = pd.to_numeric(df_demo['xgb_score'], errors='coerce')
        df_demo['gnn_score'] = pd.to_numeric(df_demo['gnn_score'], errors='coerce')
        # Store original fraud status as 0/1 for metric calculations
        df_demo['isFraud_numeric'] = df_demo[config.TARGET_COL] # Keep numeric version

        return df_demo
    except FileNotFoundError as e:
        st.error(f"Error loading data/prediction file: {e}. Ensure all necessary CSV files exist.")
        st.error(f"Looked for:\n- {trans_path}\n- {id_path}\n- {xgb_pred_path}\n- {gnn_pred_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        return pd.DataFrame()
#  Color Mapping 
def probability_to_color(prob):
    # ... (same function as before) ...
    if prob is None or pd.isna(prob):
        return 'background-color: #B0B0B0; color: #B0B0B0'
    try:
        cmap = cm.RdYlGn_r
        norm = mcolors.Normalize(vmin=0, vmax=1)
        rgba = cmap(norm(prob))
        hex_color = mcolors.rgb2hex(rgba)
        text_color = 'white' if mcolors.rgb_to_hsv(rgba[:3])[2] < 0.5 else 'black'
        return f'background-color: {hex_color}; color: {text_color}'
    except Exception as e:
        print(f"Color mapping error: {e}")
        return 'background-color: white'

#  Row Styling for Actual Fraud 
def highlight_fraud(row):
    """Applies background style to rows that are actual fraud."""
    color = '#FFDDDD'
    if 'isFraud_numeric' in row and row['isFraud_numeric'] == 1:
        return [f'background-color: {color}'] * len(row)
    else:
        return [''] * len(row)

#  Agreement Flag 
def agreement_flag(row, threshold=0.5):
    # ... (same function as before, uses gnn_score_calibrated if exists) ...
    gnn_col = 'gnn_score_calibrated' if 'gnn_score_calibrated' in row else 'gnn_score'
    xgb_pred = row['xgb_score'] > threshold if pd.notnull(row['xgb_score']) else None
    gnn_pred = row[gnn_col] > threshold if pd.notnull(row[gnn_col]) else None
    if xgb_pred is None or gnn_pred is None: return "❓"
    elif xgb_pred == gnn_pred: return "✅✅" if not xgb_pred else "🚨🚨"
    elif not xgb_pred and gnn_pred: return "✅🚨"
    else: return "🚨✅"

#  Streamlit App Layout 
st.set_page_config(layout="wide")
st.title("Fraud Detection Demo: XGBoost vs. GNN")
st.caption("Showing sample test data. Click 'Score' to reveal pre-calculated model scores.")

#  Initialize Session State 
if 'scores_calculated' not in st.session_state:
    st.session_state.scores_calculated = False
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.50

#  Load Data 
df_demo_raw = load_demo_data_with_predictions()

if df_demo_raw.empty:
    st.error("Failed to load necessary data. Cannot run demo.")
    st.stop()

#  Button to trigger scoring display 
if st.button("Score Transactions", key="score_button"):
    st.session_state.scores_calculated = True
    st.rerun() # Rerun needed to update display logic

st.divider()

#  Data Preparation & Filtering 
st.subheader("Sample Test Transactions")
df_display_full = df_demo_raw.copy()

#  Prepare columns based on whether scores are shown 
xgb_score_col = 'xgb_score'
gnn_score_col = 'gnn_score'
gnn_score_col_calibrated = 'gnn_score_calibrated'
xgb_display_name = 'XGB Score'
gnn_display_name = 'GNN Score'

if st.session_state.scores_calculated:
    # Calibrate GNN Score
    # ... (calibration logic creating gnn_score_col_calibrated) ...
    xgb_mean = df_display_full[xgb_score_col].mean()
    xgb_std = df_display_full[xgb_score_col].std()
    gnn_mean = df_display_full[gnn_score_col].mean()
    gnn_std = df_display_full[gnn_score_col].std()
    if xgb_std > 0 and gnn_std > 0:
        df_display_full[gnn_score_col_calibrated] = (((df_display_full[gnn_score_col] - gnn_mean) / gnn_std) * xgb_std) + xgb_mean
        df_display_full[gnn_score_col_calibrated] = df_display_full[gnn_score_col_calibrated].clip(0, 1)
        gnn_col_source_for_display = gnn_score_col_calibrated # Use calibrated for display
        st.caption("GNN scores calibrated to match XGBoost score distribution (mean/std) on this sample.")
    else:
        st.warning("Could not calibrate GNN scores (zero std dev). Displaying raw scores.")
        df_display_full[gnn_score_col_calibrated] = df_display_full[gnn_score_col] # Fallback
        gnn_col_source_for_display = gnn_score_col # Use raw for display

    # Calculate Score Difference
    df_display_full['Score Diff'] = abs(df_display_full[gnn_col_source_for_display] - df_display_full[xgb_score_col])
    # Calculate Agreement Flag
    df_display_full['Agreement'] = df_display_full.apply(agreement_flag, axis=1, threshold=st.session_state.threshold)

    # Point to actual score columns for styling/formatting
    xgb_col_for_style = xgb_score_col
    gnn_col_for_style = gnn_col_source_for_display

else:
    # Create placeholder columns with default values for initial display
    df_display_full[xgb_display_name] = 0.5
    df_display_full[gnn_display_name] = 0.5
    df_display_full['Score Diff'] = np.nan
    df_display_full['Agreement'] = ""
    # Point styling/formatting to these placeholder names
    xgb_col_for_style = xgb_display_name
    gnn_col_for_style = gnn_display_name

#  Format Timestamp, Amount, isFraud (always) 
# ... (formatting logic using df_display_full) ...
if config.TIMESTAMP_COL in df_display_full.columns:
    min_sample_dt = df_demo_raw[config.TIMESTAMP_COL].min()
    start_date = pd.to_datetime('2024-03-15 08:00:00')
    df_display_full['TransactionDT_Orig'] = df_display_full[config.TIMESTAMP_COL]
    df_display_full[config.TIMESTAMP_COL] = start_date + pd.to_timedelta(df_display_full['TransactionDT_Orig'] - min_sample_dt, unit='s')
    df_display_full[config.TIMESTAMP_COL] = df_display_full[config.TIMESTAMP_COL].dt.strftime('%Y-%m-%d %H:%M')
if 'TransactionAmt' in df_display_full.columns:
    df_display_full['TransactionAmt'] = df_display_full['TransactionAmt'].apply(lambda x: f'$ {x:,.2f}' if pd.notnull(x) else '')
if config.TARGET_COL in df_display_full.columns:
    df_display_full[config.TARGET_COL] = df_display_full[config.TARGET_COL].apply(lambda x: '🚨' if x == 1 else '✅')


#  Order by Time 
if 'TransactionDT_Orig' in df_display_full.columns:
    df_display_full = df_display_full.sort_values(by='TransactionDT_Orig', ascending=True)

#  Select Top 20 GNN Confidence (only if scores are shown) 
if st.session_state.scores_calculated:
    df_display_full['gnn_confidence'] = abs(df_display_full[gnn_score_col] - 0.5)
    df_display_filtered = df_display_full.sort_values(by='gnn_confidence', ascending=False).head(20)
    if 'TransactionDT_Orig' in df_display_filtered.columns:
         df_display_filtered = df_display_filtered.sort_values(by='TransactionDT_Orig', ascending=True)
    st.caption("Showing 20 transactions with highest GNN confidence (raw scores furthest from 0.5).")
else:
    df_display_filtered = df_display_full

# Drop helper column if it exists
if 'TransactionDT_Orig' in df_display_filtered.columns:
    df_display_filtered = df_display_filtered.drop(columns=['TransactionDT_Orig'])


#  Prepare Final Display DataFrame and Styler 
# Create the DataFrame with the columns intended for final display
display_cols_config = getattr(config, 'DISPLAY_COLS', ['TransactionID', 'TransactionDT', 'isFraud', 'TransactionAmt'])
# Define the source columns needed for display (using original/calibrated names)
score_cols_source = [xgb_col_for_style, gnn_col_for_style, 'Score Diff', 'Agreement']
final_display_cols_source = display_cols_config + score_cols_source
final_display_cols_source_existing = [col for col in final_display_cols_source if col in df_display_filtered.columns]

# Select only the necessary columns *before* styling
df_for_styling = df_display_filtered[final_display_cols_source_existing].copy()

# Apply styling to this filtered DataFrame
styler = df_for_styling.style.apply(highlight_fraud, axis=1) # Apply row highlighting

# Chain other styles using the correct source column names
styler = styler.format({
    xgb_col_for_style: '{:.3f}',
    gnn_col_for_style: '{:.3f}',
    'Score Diff': '{:.3f}'
}, na_rep="None").apply(lambda x: x.map(probability_to_color), subset=[xgb_col_for_style, gnn_col_for_style])

#  Rename columns *after* styling is defined, just before display 
df_to_show = df_for_styling.rename(columns={
    xgb_col_for_style: 'XGB Score',
    gnn_col_for_style: 'GNN Score'
    # Add other renames if needed
})
# Get final display names after renaming
final_display_col_names = list(df_to_show.columns)

#  Display Table with Styling 
st.dataframe(
    styler, # Pass the styler object which styles the underlying df_for_styling
    column_config={ # Use column_config for renaming and final formatting
        xgb_col_for_style: st.column_config.NumberColumn("XGB Score", format="%.3f"),
        gnn_col_for_style: st.column_config.NumberColumn("GNN Score", format="%.3f"),
        'Score Diff': st.column_config.NumberColumn("Score Diff", format="%.3f"),
        # Add configs for other columns if needed
    },
    column_order=final_display_col_names, # Ensure correct order
    use_container_width=True,
    hide_index=True
)

st.caption("Score cell color indicates predicted fraud probability (Green=Low, Red=High). Grey indicates missing score. Light red row background indicates actual fraud.")

st.divider()

#  Threshold Slider and Dynamic CM 
if st.session_state.scores_calculated:
    st.subheader("Classification Threshold Analysis (on displayed data)")
    new_threshold = st.slider(
        "Adjust Classification Threshold:",
        min_value=0.01, max_value=0.99, value=st.session_state.threshold, step=0.01,
        key="threshold_slider"
    )
    if new_threshold != st.session_state.threshold:
        st.session_state.threshold = new_threshold
        st.rerun()

    # Use df_display_filtered for metrics calculation (contains isFraud_numeric and original score names)
    df_metrics = df_display_filtered.copy()
    df_metrics['xgb_pred_class'] = (df_metrics[xgb_score_col] > st.session_state.threshold).astype(int)
    df_metrics['gnn_pred_class'] = (df_metrics[gnn_col_source_for_display] > st.session_state.threshold).astype(int)

    valid_xgb = df_metrics.dropna(subset=['isFraud_numeric', xgb_score_col])
    valid_gnn = df_metrics.dropna(subset=['isFraud_numeric', gnn_col_source_for_display])

    col1, col2 = st.columns(2)
    with col1:
        st.write("**XGBoost Confusion Matrix**")
        if not valid_xgb.empty:
            cm_xgb = confusion_matrix(valid_xgb['isFraud_numeric'], valid_xgb['xgb_pred_class'], labels=[0, 1])
            st.dataframe(pd.DataFrame(cm_xgb, index=['True Neg', 'True Fraud'], columns=['Pred Neg', 'Pred Fraud']))
        else:
            st.write("No valid XGBoost scores.")
    with col2:
        st.write("**GNN Confusion Matrix**")
        if not valid_gnn.empty:
            cm_gnn = confusion_matrix(valid_gnn['isFraud_numeric'], valid_gnn['gnn_pred_class'], labels=[0, 1])
            st.dataframe(pd.DataFrame(cm_gnn, index=['True Neg', 'True Fraud'], columns=['Pred Neg', 'Pred Fraud']))
        else:
            st.write("No valid GNN scores.")