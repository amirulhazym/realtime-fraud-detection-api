# --- demo.py ---
import streamlit as st
import pandas as pd
import numpy as np # Often needed by sklearn/xgboost
import requests
import joblib
import shap
import matplotlib.pyplot as plt # For displaying SHAP plot

# --- Configuration ---
API_ENDPOINT_URL = "https://ino023h7ib.execute-api.ap-southeast-5.amazonaws.com/predict" # visit github for latest endpoint URL

MODEL_FILE_PATH = "best_fraud_pipeline.joblib" 

# --- Page Configuration (Optional but good practice) ---
st.set_page_config(
    page_title="Real-Time Fraud Detection Demo",
    page_icon="🛡️",
    layout="wide"
)

# --- Load Model for SHAP (Error handling is good) ---
try:
    pipeline_for_shap = joblib.load(MODEL_FILE_PATH)
    st.sidebar.success(f"SHAP Model '{MODEL_FILE_PATH}' loaded successfully!")
except FileNotFoundError:
    pipeline_for_shap = None
    st.sidebar.error(f"SHAP Model file '{MODEL_FILE_PATH}' not found. SHAP plots will be unavailable.")
except Exception as e:
    pipeline_for_shap = None
    st.sidebar.error(f"Error loading SHAP model: {e}")

# --- Application Title ---
st.title("🚀 Real-Time Fraud Detection API Demo")
st.markdown("This demo interacts with a live AWS Lambda API for fraud predictions and shows local SHAP explanations.")

# --- Input Features Form ---
st.sidebar.header("Transaction Features Input:")

# Use two columns for better layout in the sidebar
col1, col2 = st.sidebar.columns(2)

with col1:
    step = st.number_input("Step (e.g., hour of day)", min_value=1, max_value=744, value=10, step=1)
    amount = st.number_input("Amount", min_value=0.0, value=5000.0, format="%.2f")
    oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0, value=20000.0, format="%.2f")
    newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0, value=15000.0, format="%.2f")
    type_CASH_OUT = st.selectbox("Type: CASH_OUT", [0, 1], index=1) # Example default
    type_DEBIT = st.selectbox("Type: DEBIT", [0, 1], index=0)

with col2:
    oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, value=1000.0, format="%.2f")
    newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, value=6000.0, format="%.2f")
    amt_ratio_orig = st.number_input("Amount Ratio Origin (amount/oldbalanceOrg)", min_value=0.0, value=0.25, format="%.3f", help="Calculated as amount / oldbalanceOrg. If oldbalanceOrg is 0, use a sensible default or handle in FE.")
    type_PAYMENT = st.selectbox("Type: PAYMENT", [0, 1], index=0)
    type_TRANSFER = st.selectbox("Type: TRANSFER", [0, 1], index=0)

# Create the feature dictionary for the API
# IMPORTANT: The order of features in the DataFrame created for SHAP later
# MUST match the order the pipeline was trained on.
# The API expects a dictionary, so order there is less critical, but the Pydantic model defines fields.
feature_input_dict = {
    "step": step,
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": oldbalanceDest,
    "newbalanceDest": newbalanceDest,
    "type_CASH_OUT": type_CASH_OUT,
    "type_DEBIT": type_DEBIT,
    "type_PAYMENT": type_PAYMENT,
    "type_TRANSFER": type_TRANSFER,
    "amt_ratio_orig": amt_ratio_orig
    # Add other one-hot encoded 'type_' features if your model expects them
    # e.g., if 'type_CASH_IN' was NOT dropped and is a feature.
}

# --- Prediction Button and API Call ---
if st.sidebar.button("Get Fraud Prediction", type="primary"):
    st.subheader("API Prediction Result:")
    with st.spinner("Calling Fraud Detection API..."):
        try:
            response = requests.post(API_ENDPOINT_URL, json=feature_input_dict, timeout=30) # 30 second timeout
            response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
            api_result = response.json()

            st.success("API Call Successful!")
            st.write("Prediction Label:", api_result.get("prediction_label", "N/A"))
            st.write("Is Fraud?:", api_result.get("is_fraud", "N/A"))
            st.write("Probability of Fraud:", api_result.get("probability_fraud", "N/A"))
            st.json(api_result) # Display raw JSON response

            # --- SHAP Explanations (if model loaded) ---
            if pipeline_for_shap:
                st.subheader("Transaction Feature Importance (SHAP):")
                with st.spinner("Calculating SHAP values..."):
                    try:
                        # Create DataFrame from input, ensuring correct column order and types
                        # This order MUST match X_train_fe when the model was trained
                        expected_feature_order = [
                            'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                            'oldbalanceDest', 'newbalanceDest', 'type_CASH_OUT', 
                            'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER', 'amt_ratio_orig'
                        ]
                        input_df_for_shap = pd.DataFrame([feature_input_dict])[expected_feature_order]

                        # If your pipeline does preprocessing (e.g., scaling, encoding from raw inputs),
                        # you might need to pass a DataFrame with raw features before preprocessing.
                        # However, if 'pipeline_for_shap' expects already feature-engineered input
                        # (like X_train_fe), then input_df_for_shap should match that structure.

                        # Assuming 'pipeline_for_shap' is your scikit-learn pipeline object
                        # The SHAP explainer works best with the underlying model, not the full pipeline if it has complex transformers.
                        # Access the final model step (e.g., XGBoost model)
                        model_for_shap = pipeline_for_shap.steps[-1][1] # e.g., pipeline.named_steps['xgboost']

                        # SHAP explainer
                        # For tree models like XGBoost, TreeExplainer is efficient
                        explainer = shap.TreeExplainer(model_for_shap)
                        
                        # If your pipeline's preprocessing steps are simple and SHAP can handle them,
                        # you might try explaining the whole pipeline on the raw input.
                        # However, usually, you explain the final model on the data as it sees it.
                        # This requires your `input_df_for_shap` to be in the *transformed* state
                        # if your pipeline's preprocessor is complex.
                        # If your pipeline already includes feature engineering and one-hot encoding,
                        # and `pipeline_for_shap.predict()` works on `input_df_for_shap`, then
                        # `shap_values = explainer.shap_values(input_df_for_shap)` might work.
                        
                        # For simplicity here, we assume `input_df_for_shap` is what the model_for_shap expects.
                        # You might need to call `pipeline_for_shap.transform(raw_input_df)` if your
                        # model_for_shap expects transformed data and your pipeline_for_shap has a transform method.

                        shap_values = explainer.shap_values(input_df_for_shap)

                        # Display SHAP plot (Waterfall for single prediction is good)
                        st.markdown("##### SHAP Waterfall Plot:")
                        fig_waterfall, ax_waterfall = plt.subplots()
                        shap.waterfall_plot(shap.Explanation(values=shap_values[0], # for first instance
                                                             base_values=explainer.expected_value,
                                                             data=input_df_for_shap.iloc[0],
                                                             feature_names=input_df_for_shap.columns.tolist()),
                                            show=False)
                        st.pyplot(fig_waterfall)

                        # Force plot (another common one for single predictions)
                        # st.markdown("##### SHAP Force Plot:")
                        # st.pyplot(shap.force_plot(explainer.expected_value, shap_values[0,:], input_df_for_shap.iloc[0,:], matplotlib=True, show=False))
                        # Note: force_plot might need `shap.initjs()` in some environments if not rendering directly to matplotlib.

                    except Exception as e_shap:
                        st.error(f"Error generating SHAP plot: {e_shap}")
                        st.error("Ensure the input data format matches what the SHAP model expects.")
            else:
                st.warning("SHAP model not loaded, so explanations are not available.")

        except requests.exceptions.RequestException as e_api:
            st.error(f"API Call Failed: {e_api}")
        except Exception as e_general:
            st.error(f"An unexpected error occurred: {e_general}")
else:
    st.info("Enter transaction features in the sidebar and click 'Get Fraud Prediction'.")

# --- Footer (Optional) ---
st.markdown("---")
st.markdown("Developed by Amirulhazym | Portfolio Project")