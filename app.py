import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from utils.preprocess import make_test_sequences
from utils.predict import predict_rul
import os
import tf_keras

st.set_page_config(page_title="RUL Predictor", layout="wide")

st.markdown("# Turbofan Engine - Remaining Useful Life Predictor")

col1, col2, col3, col4 = st.columns(4)
with col4:
    st.link_button("by-Vikas Dubey", url="https://www.linkedin.com/in/vikas-dubey-27719831a/", type="tertiary")

st.divider()
st.markdown("## Trained on NASA Turbofan Jet Engine Data Set")
st.link_button("View Dataset", "https://www.kaggle.com/datasets/behrad3d/nasa-cmaps/code", type="primary")

columns_names = ['unit_number', 'Operation_cycle', 'Operational_settings_1',
                 'Operational_settings_2', 'Operational_settings_3'] + \
                [f'Sensor_measurement_{i}' for i in range(1, 22)]

df_path = os.path.join("dataset", "train_FD001.txt")
df_train = pd.read_csv(df_path, sep='\s+', header=None, names=columns_names, dtype=float)
st.dataframe(df_train.head(), hide_index=True)

st.markdown("""
### Useful vs Redundant Sensor Data

Not all 21 sensors in this dataset are equally informative. Some sensors remain **almost constant** throughout the entire engine lifecycle, meaning they carry little to no signal about engine degradation.

Keeping these flat sensors would:
- **Increase model dimensionality** unnecessarily
- **Add noise** that can confuse the model
- **Slow down training** without any accuracy benefit

So during preprocessing, sensors with **near-zero variance** were identified and removed. Only the sensors that show a **clear trend or pattern** over the engine's life cycles were kept for training.

The images below show the difference — **constant sensors** (removed) vs **active sensors** (used for training).
""")
col1, col2 = st.columns(2)
with col1:
    st.image(os.path.join("assets", "constant_sensor.png"), caption="Constant Sensors", use_container_width=True)
with col2:
    st.image(os.path.join("assets", "active_sensor.png"), caption="Active Sensors", use_container_width=True)
st.divider()
st.markdown("### About Trained Models")
tab1, tab2 = st.tabs(["Model 1", "Model 2"])
with tab1:
    st.markdown("""
**Model 1** is trained on `Window_size = 7` and `Horizon_size = 1`.

- Uses the last **7 cycles** of sensor data to predict the next RUL value.
- Best suited for **short-term pattern detection** where recent behavior matters most.
- Lightweight and **faster to run**, ideal for quick health checks.
- Works well when engine data has **less historical depth** available.
""")
    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join("assets", "Window_7.png"), caption="Graph btw loss and val_loss", width=500)
    with col2:
        df_1 = pd.DataFrame({"Evaluation Metric": ['MAE', 'MSE', 'RMSE', 'MAPE', 'MASE'],
                             "Value":  [13.397884, 338.47043, 18.397566, 23.459606, 0.29215652]})
        st.dataframe(df_1,hide_index= True)
with tab2:
    st.markdown("""
**Model 2** is trained on `Window_size = 30` and `Horizon_size = 1`.

- Uses the last **30 cycles** of sensor data to predict the next RUL value.
- Captures **longer degradation trends**, making predictions more stable over time.
- Better suited for engines with **rich historical data** available.
- Generally more **accurate but requires more data** — at least 30 cycles per engine.
""")
    col3, col4 = st.columns(2)
    with col3:
        st.image(os.path.join("assets", "Window_30.png"), caption="Graph btw loss and val_loss", width=500)
    with col4:
        df_2 = pd.DataFrame({"Evaluation Metric": ['MAE', 'MSE', 'RMSE', 'MAPE', 'MASE'],
                             "Value":  [10.60941, 214.4237, 14.643213, 15.257753, 0.23309506]})
        st.dataframe(df_2,hide_index= True)

st.divider()
st.markdown("## Try it out!")

# --- Initialize session state ---
if "model" not in st.session_state:
    st.session_state.model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "WINDOW_SIZE" not in st.session_state:
    st.session_state.WINDOW_SIZE = None
if "HORIZON" not in st.session_state:
    st.session_state.HORIZON = None

st.markdown("### Step 01: Select Model")
left, right, a, b = st.columns(4)

if left.button("Model 1"):
    st.session_state.WINDOW_SIZE = 7
    st.session_state.HORIZON = 1
    st.session_state.model = tf_keras.models.load_model('model/model_w7.h5')
    with open('scaler/scaler.pkl', "rb") as f:
        st.session_state.scaler = pickle.load(f)

if right.button("Model 2"):
    st.session_state.WINDOW_SIZE = 30
    st.session_state.HORIZON = 1
    st.session_state.model = tf_keras.models.load_model('model/model_w30.h5')
    with open('scaler/scaler.pkl', "rb") as f:
        st.session_state.scaler = pickle.load(f)

if st.session_state.model is not None:
    st.success(f"✅ Model loaded — Window Size = {st.session_state.WINDOW_SIZE}, Horizon = {st.session_state.HORIZON}")
else:
    st.warning("No model selected yet.")
st.divider()
# --- File Upload ---
st.markdown("### Step 02: Upload your Data")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

X_test_scaled = None

if uploaded_file is not None and st.session_state.model is not None:
    test_data = pd.read_csv(uploaded_file)

    if len(test_data) < st.session_state.WINDOW_SIZE:
        st.error(f"Need at least {st.session_state.WINDOW_SIZE} cycles")
        st.stop()

    st.dataframe(test_data.head(), hide_index=True)

    feature_cols = [col for col in test_data.columns
                    if col not in ['unit_number', 'Operation_cycle']]

    X_test = make_test_sequences(test_data, st.session_state.WINDOW_SIZE, feature_cols)
    X_test_scaled = st.session_state.scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])
    ).reshape(X_test.shape)
st.divider()
# --- Predict ---
st.markdown("### Step 03: Get Prediction")

if st.button("Predict RUL", type="primary"):
    if X_test_scaled is None or st.session_state.model is None:
        st.warning("Please select a model and upload a CSV file first.")
    else:
        predictions = predict_rul(st.session_state.model, X_test_scaled)

        st.subheader("📊 Prediction Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean RUL", f"{predictions.mean():.2f} cycles")
        col2.metric("Min RUL", f"{predictions.min():.2f} cycles")
        col3.metric("Max RUL", f"{predictions.max():.2f} cycles")

        st.subheader("📈 RUL Over Time")
        result_df = pd.DataFrame(predictions, columns=[f"Step+{i+1}" for i in range(st.session_state.HORIZON)])

        # Add unit_number column to result_df for alignment
        unit_numbers = test_data["unit_number"].values[st.session_state.WINDOW_SIZE - 1:]
        
        # Trim to match prediction length (safety fix)
        min_len = min(len(result_df), len(unit_numbers))
        result_df = result_df.iloc[:min_len].reset_index(drop=True)
        unit_numbers = unit_numbers[:min_len]
        result_df["unit_number"] = unit_numbers

        st.line_chart(result_df.drop(columns=["unit_number"]))

        # --- Per Engine Analysis ---
        st.subheader("🔩 Per Engine Analysis")
        available_units = sorted(result_df["unit_number"].unique())
        selected_unit = st.selectbox("Select Engine Unit", available_units)

        unit_preds = result_df[result_df["unit_number"] == selected_unit].drop(columns=["unit_number"])

        if unit_preds.empty:
            st.warning(f"No predictions found for Engine Unit {selected_unit}.")
        else:
            st.line_chart(unit_preds)

        # --- Critical Engine Alerts ---
        st.subheader("⚠️ Critical Engines")
        threshold = st.slider("RUL Alert Threshold (cycles)", 10, 100, 30)
        pred_values = result_df.drop(columns=["unit_number"]).values
        critical_mask = pred_values < threshold
        critical_count = critical_mask.sum()

        if critical_count > 0:
            st.error(f"{critical_count} engine windows below threshold!")

            # Show which units are critical
            critical_units = result_df[result_df.drop(columns=["unit_number"]).lt(threshold).any(axis=1)]["unit_number"].unique()
            st.write("**Critical Engine Units:**", list(critical_units))
        else:
            st.success("All engines are healthy!")

        # --- Download (always available) ---
        st.subheader("⬇️ Download Results")
        # Reorder and rename for clean download
        download_df = result_df[["unit_number"] + [col for col in result_df.columns if col != "unit_number"]].copy()
        download_df = download_df.rename(columns={f"Step+{i+1}": "Predicted_RUL" for i in range(st.session_state.HORIZON)})
        
        st.download_button(
            label="Download Full Predictions CSV",
            data=download_df.to_csv(index=False),
            file_name="rul_predictions.csv",
            mime="text/csv",
            type="primary"

        )
