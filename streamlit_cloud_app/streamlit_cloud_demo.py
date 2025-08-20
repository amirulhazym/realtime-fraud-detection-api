# --- streamlit_cloud_demo.py ---
# Note: Most of the code is commented out as a moving from EC2 to Streamlit Cloud;
# cost-risk management & backup strategy

import streamlit as st
import pandas as pd
import numpy as np # needed by sklearn/xgboost
import requests
# import joblib
# import shap
# import matplotlib.pyplot as plt
import plotly.express as px # ### MODIFIED FOR STREAMLIT CLOUD ### - Keep if to try display any data with plotly, but not for SHAP since SHAP is removed

# --- Configuration ---
API_ENDPOINT_URL = "https://zeir21qzal.execute-api.ap-southeast-5.amazonaws.com/predict" # Make sure this is correct
# MODEL_FILE_PATH = "best_fraud_pipeline.joblib" # 
APP_VERSION = "1.1.0-cloud" # Example version for cloud demo

# --- Page Configuration (MUST BE FIRST) ---
st.set_page_config(
    page_title="Real-Time Fraud Detection (Cloud)", 
    page_icon="icon_logo.ico", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Modern UI (Keep as is, it's UI styling) ---
st.markdown("""
<style>
    body {
        /* background: linear-gradient(to right, #1a1c2a, #1f2233); */ 
        color: #f0f2f6; 
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button { 
        background-color: #4CAF50 !important; 
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
        border: none !important; 
    }
    .stButton>button:hover {
        background-color: #45a049 !important; 
    }
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div { 
        border-radius: 8px !important;
        border: 1px solid #4CAF50 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Header with Logo (Keep as is, or ensure logo URL is accessible) ---
try:
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 15px; padding-bottom: 15px;'>
      <h1 style='margin-bottom: 0px; color: #4CAF50;'>Real-Time Fraud Detection System</h1>
    </div>
    """, unsafe_allow_html=True)
except FileNotFoundError: # try-except might not be needed if using a web URL for the logo
    st.title("🚀 Real-Time Fraud Detection System") 

# ### MODIFIED FOR STREAMLIT CLOUD ### 
st.markdown("#### 🔍 Instantly detect potentially fraudulent transactions using our AI-powered API.")
st.markdown("---")


# ### MODIFIED FOR STREAMLIT CLOUD ### - Remove SHAP model loading function and call
# @st.cache_data 
# def load_model(path):
#     try:
#         return joblib.load(path)
#     except FileNotFoundError:
#         return None 
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None

# pipeline_for_shap = load_model(MODEL_FILE_PATH) 
pipeline_for_shap = None ### MODIFIED FOR STREAMLIT CLOUD ### - Set to None explicitly


# --- Sidebar Input Form ---
with st.sidebar:
    st.header("📌 Transaction Simulator")
    st.markdown("Enter details and click **Predict** below:")

    # ### MODIFIED FOR STREAMLIT CLOUD ### - Change SHAP model status message
    st.info("SHAP explanations are not available in this cloud-hosted demo version.")
    st.markdown("---")

    # Group inputs into collapsible sections (UI structure V3.0)
    with st.expander("Transaction Details", expanded=True):
        step = st.number_input("Step (Hour of Day, 1-744)", 1, 744, 10, help="Time step of the transaction.")
        amount = st.number_input("Amount (RM)", 0.0, None, 5000.0, format="%.2f", help="Monetary value of the transaction.")
        transaction_type_options = ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER", "CASH_IN"] 
        transaction_type = st.selectbox("Transaction Type", transaction_type_options, index=0, help="Select the type of transaction.")

    with st.expander("Account Balances (Before & After)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            oldbalanceOrg = st.number_input("Sender Old Balance", 0.0, None, 20000.0, format="%.2f")
            newbalanceOrig = st.number_input("Sender New Balance", 0.0, None, 15000.0, format="%.2f")
        with col2:
            oldbalanceDest = st.number_input("Receiver Old Balance", 0.0, None, 1000.0, format="%.2f")
            newbalanceDest = st.number_input("Receiver New Balance", 0.0, None, 6000.0, format="%.2f")
    st.markdown("---")
    
    if oldbalanceOrg > 0:
        amt_ratio_orig = amount / oldbalanceOrg
    else:
        amt_ratio_orig = 1.0 if amount > 0 else 0.0 
    
    st.info(f"Calculated 'Amount Ratio Origin': {amt_ratio_orig:.4f}")

    type_CASH_OUT = 1 if transaction_type == "CASH_OUT" else 0
    type_DEBIT = 1 if transaction_type == "DEBIT" else 0
    type_PAYMENT = 1 if transaction_type == "PAYMENT" else 0
    type_TRANSFER = 1 if transaction_type == "TRANSFER" else 0
    
    feature_input_dict_for_api = {
        "step": int(step), 
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
    }
    
    # ### MODIFIED FOR STREAMLIT CLOUD ### - expected_feature_order_for_shap is not needed since SHAP is removed
    # expected_feature_order_for_shap = [
    #     'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
    #     'oldbalanceDest', 'newbalanceDest', 'type_CASH_OUT', 
    #     'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER', 'amt_ratio_orig'
    # ]

    predict_button = st.button("🔮 Predict Fraud Risk", type="primary", use_container_width=True, help="Click to submit transaction for fraud analysis.")
    
    st.markdown("---") # ### MODIFIED FOR STREAMLIT CLOUD ### - Moved from original sidebar status messages
    st.caption(f"App Version: {APP_VERSION} | Developed by Amirulhazym")


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
        except Exception as e: # Catching broader exceptions during API call
            st.error(f"❌ An unexpected error occurred during API call: {str(e)}")
            api_result = None
            api_call_successful = False

    if api_call_successful and api_result:
        st.success("✅ API Prediction Received Successfully!")
        
        is_fraud_api = api_result.get("is_fraud", False)
        prob_fraud_api_str = str(api_result.get("probability_fraud", "0.0"))
        try:
            prob_fraud_api = float(prob_fraud_api_str.replace('f', '').replace(',', ''))
        except ValueError:
            prob_fraud_api = 0.0 
            st.warning(f"Could not parse fraud probability '{prob_fraud_api_str}' from API. Defaulting to 0.0.")

        prediction_label_api = "FRAUDULENT" if is_fraud_api else "NOT FRAUDULENT"

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
            display_dict = {
                "Amount": f"{amount:,.2f}",
                "Type": transaction_type,
                "Origin Old Bal.": f"{oldbalanceOrg:,.2f}",
                "Origin New Bal.": f"{newbalanceOrig:,.2f}",
                "Ratio (Amt/OrigBal)": f"{amt_ratio_orig:.3f}"
            }
            st.json(display_dict)
        
        st.caption("Full API Response:")
        st.json(api_result)
        st.markdown("---")

        # ### MODIFIED FOR STREAMLIT CLOUD ### - Entire SHAP explanation block removed/commented
        # if pipeline_for_shap:
        #     st.subheader("📊 Model Explanation (Feature Importance via SHAP)")
        #     st.markdown("This plot shows how each feature value for *this specific transaction* pushed the model's prediction score away from the average. Red bars push towards fraud, blue bars push away from fraud.")
        #     with st.spinner("⚙️ Calculating SHAP values..."):
        #         try:
        #             input_df_for_shap = pd.DataFrame([feature_input_dict_for_api])[expected_feature_order_for_shap]
        #             model_for_shap = pipeline_for_shap.steps[-1][1] 
        #             explainer = shap.TreeExplainer(model_for_shap)
        #             shap_values_raw = explainer.shap_values(input_df_for_shap) 
        #             shap_values_for_instance = shap_values_raw[0]
        #             shap_df = pd.DataFrame({
        #                 'Features': input_df_for_shap.columns,
        #                 'SHAP_Values': shap_values_for_instance
        #             })
        #             shap_df['abs_SHAP_Values'] = shap_df['SHAP_Values'].abs()
        #             shap_df = shap_df.sort_values(by='abs_SHAP_Values', ascending=True)
        #             fig = px.bar(
        #                 shap_df, 
        #                 x='SHAP_Values', 
        #                 y='Features', 
        #                 orientation='h', 
        #                 title="Feature Contributions to Prediction Score",
        #                 color='SHAP_Values',
        #                 color_continuous_scale=px.colors.diverging.RdBu_r, 
        #                 labels={'SHAP_Values': 'SHAP Value (Impact on Model Output)', 'Features': 'Transaction Feature'}
        #             )
        #             fig.update_layout(
        #                 yaxis_title=None, 
        #                 coloraxis_colorbar_title_text='Impact'
        #             )
        #             st.plotly_chart(fig, use_container_width=True)
        #         except Exception as e_shap_plot:
        #             st.error(f"❌ Error generating SHAP plot: {str(e_shap_plot)}")
        #             st.caption("This can happen if input data format is incompatible with the SHAP model, or due to version conflicts.")
        # else:
        #     st.info("SHAP explanations are unavailable because the local model was not loaded.") # This can be removed or kept with pipeline_for_shap being None

    elif predict_button: 
        st.warning("⚠️ Could not retrieve prediction due to API error. Please check error messages above.")

else: 
    st.info("ℹ️ Enter transaction details in the sidebar and click **Predict Fraud Risk** to start the analysis.")

# --- Footer ---
st.markdown("---")
st.markdown("Developed by **Amirulhazym** | Powered by **Streamlit** & **AWS Lambda**")
# st.caption(f"App Version: 1.0.0") # Already have APP_VERSION at top, keep for original ec2 instance code