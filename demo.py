# --- demo.py (Based on YOUR provided UI, with SHAP section reviewed) ---
import streamlit as st
import pandas as pd
import numpy as np # Often needed by sklearn/xgboost
import requests
import joblib
import shap
import matplotlib.pyplot as plt # Still needed if you might switch back to shap.plots.waterfall
import plotly.express as px

# --- Page Configuration (MUST BE FIRST) ---
# For icons, you typically need to serve them or have them in a known path.
# For now, let's use an emoji or keep it simple if direct image URLs are tricky in Streamlit for page_icon.
# You'd usually place 'icon_logo.ico' and 'main_logo.png' in the same directory as demo.py
# or provide a full URL if they are hosted online.
try:
    st.set_page_config(
        page_title="Real-Time Fraud Detection",
        page_icon="🛡️", # Using an emoji as a fallback for simplicity now
        # page_icon="icon_logo.ico", # This would require icon_logo.ico to be in the root of your app
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e: # Fallback if local icon file causes issues on some environments
    st.set_page_config(
        page_title="Real-Time Fraud Detection",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )


# --- Custom CSS for Modern UI (from your code) ---
st.markdown("""
<style>
    body {
        /* background: linear-gradient(to right, #1a1c2a, #1f2233); */ /* Might be too dark with Streamlit's default theme elements */
        color: #f0f2f6; /* Ensure this works with Streamlit's base theme or a dark theme */
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button { /* This targets all Streamlit buttons */
        background-color: #4CAF50 !important; /* Added !important for override */
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
        border: none !important; /* Remove default border */
    }
    .stButton>button:hover {
        background-color: #45a049 !important; /* Darker shade on hover */
    }

    /* Targeting specific buttons if needed by adding a class in st.button("label", class_name="my-button")
    .my-button button { ... } */

    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div { /* Adjusted selector for selectbox */
        border-radius: 8px !important;
        border: 1px solid #4CAF50 !important;
        /* color: #ffffff; */ /* Text color inside input might need to be default Streamlit for theme compatibility */
        /* background-color: #2d3139; */ /* Be careful with background colors, might clash with Streamlit themes */
    }
    /* Ensure sidebar itself has a compatible background if you change element backgrounds */
    /* div[data-testid="stSidebar"] > div:first-child { background-color: #1f2233; } */
</style>
""", unsafe_allow_html=True)

# --- Header with Logo ---
# For local images, ensure they are in the same directory as demo.py
# To make it more robust, consider hosting images online (e.g., GitHub raw link) or base64 encoding.
# For now, assuming 'main_logo.png' is in the same directory as demo.py.
try:
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 15px; padding-bottom: 15px;'>
      <img src='https://raw.githubusercontent.com/amirulhazym/realtime-fraud-detection-api/main/main_logo.png' width='50' style='border-radius: 5px;'> <!-- Example: GitHub raw link -->
      <h1 style='margin-bottom: 0px; color: #4CAF50;'>Real-Time Fraud Detection System</h1>
    </div>
    """, unsafe_allow_html=True)
except FileNotFoundError:
    st.title("🚀 Real-Time Fraud Detection System") # Fallback if logo not found

st.markdown("#### 🔍 Instantly detect potentially fraudulent transactions using AI-powered insights and SHAP explanations.")
st.markdown("---")


# --- API and Model Configuration ---
API_ENDPOINT_URL = "https://zeir21qzal.execute-api.ap-southeast-5.amazonaws.com/predict" # Make sure this is correct
MODEL_FILE_PATH = "best_fraud_pipeline.joblib"

@st.cache_data # Cache the model loading
def load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None # Handled in the UI
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

pipeline_for_shap = load_model(MODEL_FILE_PATH)


# --- Sidebar Input Form ---
with st.sidebar:
    st.header("📌 Transaction Simulator")
    st.markdown("Enter details and click **Predict**.")

    if pipeline_for_shap is None:
        st.error(f"Local model '{MODEL_FILE_PATH}' not found. SHAP explanations will be unavailable.")
    else:
        st.success(f"Local model '{MODEL_FILE_PATH}' loaded for SHAP.")
    st.markdown("---")

    # Group inputs into collapsible sections
    with st.expander("Transaction Details", expanded=True):
        step = st.number_input("Step (Hour of Day, 1-744)", 1, 744, 10, help="Time step of the transaction.")
        amount = st.number_input("Amount (e.g., RM)", 0.0, None, 5000.0, format="%.2f", help="Monetary value of the transaction.")
        # Using a single selectbox for transaction type is more user-friendly
        transaction_type_options = ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER", "CASH_IN"] # Add all types your model was trained on
        transaction_type = st.selectbox("Transaction Type", transaction_type_options, index=0, help="Select the type of transaction.")

    with st.expander("Account Balances (Before & After)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            oldbalanceOrg = st.number_input("Origin Old Balance", 0.0, None, 20000.0, format="%.2f")
            newbalanceOrig = st.number_input("Origin New Balance", 0.0, None, 15000.0, format="%.2f")
        with col2:
            oldbalanceDest = st.number_input("Dest. Old Balance", 0.0, None, 1000.0, format="%.2f")
            newbalanceDest = st.number_input("Dest. New Balance", 0.0, None, 6000.0, format="%.2f")
    st.markdown("---")
    
    # Auto-calculate ratio - ensure this is done *before* constructing feature_input_dict for the API
    # And ensure it matches exactly how it was done in your feature engineering
    if oldbalanceOrg > 0:
        amt_ratio_orig = amount / oldbalanceOrg
    else:
        # Handle division by zero: what did your FE do?
        # Common: 0, or if amount > 0 then a large number (e.g., 999), or mean/median from training.
        # Let's assume 0 for now if amount is also 0, or 1 if amount > 0 and oldbalance is 0 (as if it's taking all of nothing + amount)
        # This needs to match your model's training!
        amt_ratio_orig = 1.0 if amount > 0 else 0.0 
    
    st.info(f"Calculated 'Amount Ratio Origin': {amt_ratio_orig:.4f}")


    # One-hot encode transaction type based on the single selectbox
    type_CASH_OUT = 1 if transaction_type == "CASH_OUT" else 0
    type_DEBIT = 1 if transaction_type == "DEBIT" else 0
    type_PAYMENT = 1 if transaction_type == "PAYMENT" else 0
    type_TRANSFER = 1 if transaction_type == "TRANSFER" else 0
    # Add other types if your model expects them, e.g. type_CASH_IN
    # Ensure these match the `expected_feature_order` later

    # Feature dictionary for API and SHAP
    # This dictionary is what your API expects.
    feature_input_dict_for_api = {
        "step": int(step), # Ensure correct types if API Pydantic model is strict
        "amount": float(amount),
        "oldbalanceOrg": float(oldbalanceOrg),
        "newbalanceOrig": float(newbalanceOrig),
        "oldbalanceDest": float(oldbalanceDest),
        "newbalanceDest": float(newbalanceDest),
        "type_CASH_OUT": int(type_CASH_OUT),
        "type_DEBIT": int(type_DEBIT),
        "type_PAYMENT": int(type_PAYMENT),
        "type_TRANSFER": int(type_TRANSFER),
        "amt_ratio_orig": float(amt_ratio_orig)
        # Add other one-hot encoded 'type_' features if your model / Pydantic model expects them
    }
    
    # IMPORTANT: Define the EXACT order of features your model pipeline was TRAINED ON for SHAP
    # This must match the columns of X_train_fe that went into pipeline_for_shap.fit()
    expected_feature_order_for_shap = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest', 'type_CASH_OUT', 
        'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER', 'amt_ratio_orig'
        # Add/remove/reorder to *exactly* match your training data columns fed to the pipeline
    ]

    # --- Prediction Button ---
    predict_button = st.button("🔮 Predict Fraud Risk", type="primary", use_container_width=True, help="Click to submit transaction for fraud analysis.")


# --- Main Area: Results ---
if predict_button:
    st.markdown("---")
    st.header("📈 Prediction & Analysis Results")
    
    with st.spinner("⏳ Analyzing transaction with AWS Lambda API..."):
        try:
            response = requests.post(API_ENDPOINT_URL, json=feature_input_dict_for_api, timeout=20)
            response.raise_for_status()
            api_result = response.json()
            api_call_successful = True
        except requests.exceptions.RequestException as e:
            st.error(f"❌ API Communication Error: {e}")
            api_result = None
            api_call_successful = False
        except Exception as e:
            st.error(f"❌ An unexpected error occurred during API call: {str(e)}")
            api_result = None
            api_call_successful = False

    if api_call_successful and api_result:
        st.success("✅ API Prediction Received Successfully!")
        
        # Extract results
        is_fraud_api = api_result.get("is_fraud", False)
        prob_fraud_api_str = str(api_result.get("probability_fraud", "0.0")) # Ensure it's a string
        try:
            # Handle potential "f6.0000" like strings or other non-float strings
            prob_fraud_api = float(prob_fraud_api_str.replace('f', '').replace(',', ''))
        except ValueError:
            prob_fraud_api = 0.0 # Default on conversion error
            st.warning(f"Could not parse fraud probability '{prob_fraud_api_str}' from API. Defaulting to 0.0.")

        prediction_label_api = "FRAUDULENT" if is_fraud_api else "NOT FRAUDULENT"

        # Display Results in columns
        res_col1, res_col2 = st.columns([2,3]) 
        with res_col1:
            st.subheader("Summary:")
            if is_fraud_api:
                st.error(f"🚨 **Status: {prediction_label_api}**")
            else:
                st.success(f"✅ **Status: {prediction_label_api}**")
            st.metric(label="Fraud Probability", value=f"{prob_fraud_api:.2%}")
            
        with res_col2:
            st.subheader("Transaction Snapshot:")
            # Display a summary of the input that was sent
            display_dict = {
                "Amount": f"{amount:,.2f}",
                "Type": transaction_type,
                "Origin Old Bal.": f"{oldbalanceOrg:,.2f}",
                "Origin New Bal.": f"{newbalanceOrig:,.2f}",
                "Ratio (Amt/OrigBal)": f"{amt_ratio_orig:.3f}"
            }
            st.json(display_dict) # Shows a neat summary of what was tested
        
        st.caption("Full API Response:")
        st.json(api_result) # Raw API response
        st.markdown("---")

        # --- SHAP Explanation (if model loaded) ---
        if pipeline_for_shap:
            st.subheader("📊 Model Explanation (Feature Importance via SHAP)")
            st.markdown("This plot shows how each feature value for *this specific transaction* pushed the model's prediction score away from the average. Red bars push towards fraud, blue bars push away from fraud.")
            with st.spinner("⚙️ Calculating SHAP values..."):
                try:
                    # Create DataFrame for SHAP using the defined order
                    input_df_for_shap = pd.DataFrame([feature_input_dict_for_api])[expected_feature_order_for_shap]
                    
                    model_for_shap = pipeline_for_shap.steps[-1][1] # Access the XGBoost model from the pipeline
                    
                    # Standard TreeExplainer initialization
                    explainer = shap.TreeExplainer(model_for_shap)
                    
                    # Get SHAP values (this is a numpy array for TreeExplainer with single output)
                    shap_values_raw = explainer.shap_values(input_df_for_shap) 
                    
                    # For your Plotly bar chart, you need SHAP values for the single instance
                    shap_values_for_instance = shap_values_raw[0] # Get the first (and only) row

                    # Create DataFrame for Plotly Express
                    shap_df = pd.DataFrame({
                        'Features': input_df_for_shap.columns,
                        'SHAP_Values': shap_values_for_instance
                    })
                    # Sort by absolute SHAP value for better visualization, or keep original order
                    shap_df['abs_SHAP_Values'] = shap_df['SHAP_Values'].abs()
                    shap_df = shap_df.sort_values(by='abs_SHAP_Values', ascending=True) # Ascending for horizontal bar

                    fig = px.bar(
                        shap_df, 
                        x='SHAP_Values', 
                        y='Features', 
                        orientation='h', 
                        title="Feature Contributions to Prediction Score",
                        color='SHAP_Values',
                        color_continuous_scale=px.colors.diverging.RdBu_r, # Red for positive, Blue for negative
                        labels={'SHAP_Values': 'SHAP Value (Impact on Model Output)', 'Features': 'Transaction Feature'}
                    )
                    fig.update_layout(
                        yaxis_title=None, # Remove y-axis title if feature names are clear
                        coloraxis_colorbar_title_text='Impact'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # If you want to try SHAP Waterfall again (more standard for single instance explanations)
                    # with st.expander("Show SHAP Waterfall Plot (Alternative View)", expanded=False):
                    #     fig_waterfall, ax_waterfall = plt.subplots()
                    #     # Create SHAP Explanation object for the plot
                    #     shap_explanation_for_plot = shap.Explanation(
                    #         values=shap_values_for_instance,
                    #         base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else None, 
                    #         data=input_df_for_shap.iloc[0].values,
                    #         feature_names=input_df_for_shap.columns.tolist()
                    #     )
                    #     shap.plots.waterfall(shap_explanation_for_plot, max_display=12, show=False)
                    #     plt.tight_layout()
                    #     st.pyplot(fig_waterfall, clear_figure=True)

                except Exception as e_shap_plot:
                    st.error(f"❌ Error generating SHAP plot: {str(e_shap_plot)}")
                    st.caption("This can happen if input data format is incompatible with the SHAP model, or due to version conflicts.")
        else:
            st.info("SHAP explanations are unavailable because the local model was not loaded.")
    elif predict_button: # Predict button was pressed, but API call failed
        st.warning("⚠️ Could not retrieve prediction due to API error. Please check error messages above.")

else: # Initial page load, before button is pressed
    st.info("ℹ️ Enter transaction details in the sidebar and click **Predict Fraud Risk** to start the analysis.")

# --- Footer ---
st.markdown("---")
st.markdown("Developed by **Amirulhazym** | Powered by **Streamlit** & **AWS Lambda**")
st.caption(f"App Version: 1.0.0") # Example version
