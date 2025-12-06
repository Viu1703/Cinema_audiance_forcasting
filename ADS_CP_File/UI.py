import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Cinema AI Forecast",
    layout="wide",
    page_icon="🎬",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING (Optional) ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
    }
    .highlight {
        color: #FF4B4B;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. LOAD SAVED ASSETS ---
@st.cache_resource
def load_assets():
    try:
        # Load the files we saved from Jupyter
        model = joblib.load('lgbm_cinema_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        threshold = joblib.load('threshold.pkl')
        return model, preprocessor, threshold
    except FileNotFoundError:
        return None, None, None

model, preprocessor, threshold = load_assets()

# --- 2. ERROR HANDLING (If files are missing) ---
if model is None:
    st.error("🚨 Critical Error: Model files not found!")
    st.markdown("""
    **How to fix this:**
    1. Open your Jupyter Notebook.
    2. Run the **save_artifacts.py** code provided earlier.
    3. Ensure `lgbm_cinema_model.pkl`, `preprocessor.pkl`, and `threshold.pkl` appear in this folder.
    4. Refresh this page.
    """)
    st.stop()

# --- 3. SIDEBAR: USER INPUTS ---
st.sidebar.header("🎥 Forecast Settings")
st.sidebar.markdown("---")

# Date Selection
selected_date = st.sidebar.date_input("Select Date", datetime.date.today())

# Theater ID (Mocking the 826 IDs from your dataset)
theater_options = [f"book_{i:05d}" for i in range(1, 11)] # Top 10 sample
theater_id = st.sidebar.selectbox("Select Theater ID", theater_options)

st.sidebar.subheader("History Data")
# In a real app, these would auto-fill from a database
lag_1 = st.sidebar.number_input(
    "Attendance Yesterday (Lag 1)", 
    min_value=0, 
    value=45,
    help="How many people visited this theater yesterday?"
)

roll7 = st.sidebar.number_input(
    "7-Day Rolling Avg", 
    min_value=0, 
    value=42,
    help="Average visitors over the last week."
)

run_btn = st.sidebar.button("🚀 Run Prediction", type="primary")

# --- 4. MAIN APP LOGIC ---
st.title("🎬 Cinema Audience Forecasting")
st.markdown("### Operational Demand Prediction System")

if run_btn:
    # A. Feature Engineering
    # We must match the EXACT columns and order used in training
    # columns = ['day', 'month', 'year', 'dow', 'lag_1', 'roll7', 'book_theater_id']
    
    input_data = pd.DataFrame({
        'day': [selected_date.day],
        'month': [selected_date.month],
        'year': [selected_date.year],
        'dow': [selected_date.weekday()], # 0=Monday, 6=Sunday
        'lag_1': [lag_1],
        'roll7': [roll7],
        'book_theater_id': [theater_id]
    })

    # B. Preprocessing
    # Use the saved ColumnTransformer to One-Hot Encode the theater ID
    try:
        X_processed = preprocessor.transform(input_data)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

    # C. Prediction (REGRESSION)
    # The error 'Booster object has no attribute predict_proba' happened 
    # because you used a Regressor. We use .predict() to get the number.
    predicted_count = model.predict(X_processed)[0]
    
    # D. Classification Logic (Manual Thresholding)
    # We convert the number to a category based on your median threshold
    is_high_traffic = predicted_count >= threshold
    
    traffic_label = "HIGH TRAFFIC" if is_high_traffic else "LOW TRAFFIC"
    traffic_color = "inverse" if is_high_traffic else "normal" # Streamlit metric colors

    # --- 5. RESULTS DISPLAY ---
    st.divider()
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Predicted Audience", 
            value=f"{int(predicted_count)} Visitors",
            delta=f"{int(predicted_count - roll7)} vs Avg"
        )

    with col2:
        st.metric(
            label="Traffic Classification", 
            value=traffic_label, 
            delta="Above Threshold" if is_high_traffic else "Below Threshold",
            delta_color=traffic_color
        )

    with col3:
        # Operational Advice
        if is_high_traffic:
            st.warning("⚠️ Action: Increase Staffing")
        else:
            st.success("✅ Action: Standard Staffing")

    # Debug / Info Section
    with st.expander("See Prediction Details"):
        st.write(f"**Model Used:** LightGBM Regressor")
        st.write(f"**Classification Threshold:** {threshold:.2f} visitors")
        st.write("**Input Data:**")
        st.dataframe(input_data)

else:
    st.info("👈 Adjust parameters in the sidebar and click 'Run Prediction'")