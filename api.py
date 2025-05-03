# --- api.py ---
import joblib
import pandas as pd
import numpy as np # Often needed by sklearn/xgboost
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field # Field for validation/examples
import uvicorn # For local running block
import os


print("Attempting to load pipeline...")
# --- Load Model Pipeline ---
# Assumes api.py is in the project root and model is too
pipeline_filename = "best_fraud_pipeline.joblib"
pipeline = None
if os.path.exists(pipeline_filename):
    try:
        # Use the pipeline saved from RandomizedSearchCV
        pipeline = joblib.load(pipeline_filename)
        print(f"Successfully loaded pipeline object: {type(pipeline)}")
        # Check if features can be accessed (depends on pipeline structure)
        # This might need adjustment based on your exact pipeline steps
        try:
             # Try accessing from the final estimator step ('xgboost')
             if hasattr(pipeline.steps[-1][1], 'feature_names_in_'):
                  expected_features_in = pipeline.steps[-1][1].feature_names_in_
                  print(f"Pipeline (estimator) expects features: {list(expected_features_in)}")
             # Fallback if the pipeline itself has the attribute (less common)
             elif hasattr(pipeline, 'feature_names_in_'):
                  expected_features_in = pipeline.feature_names_in_
                  print(f"Pipeline (direct) expects features: {list(expected_features_in)}")
             else:
                  print("Could not automatically determine expected features from pipeline object.")
                  # Fallback: Manually list based on L2.1 X_train_fe columns
                  expected_features_in = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER', 'amt_ratio_orig']
                  print(f"Manually listing expected features: {expected_features_in}")


        except AttributeError:
             print("Could not automatically determine expected features.")
             expected_features_in = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER', 'amt_ratio_orig']
             print(f"Manually listing expected features: {expected_features_in}")


    except Exception as e:
        print(f"ERROR: Failed to load pipeline from {pipeline_filename}. Error: {e}")
        # API will start but predictions will fail.
else:
    print(f"ERROR: Pipeline file '{pipeline_filename}' not found in current directory.")
    expected_features_in = [] # Define as empty if load fails


# --- Placeholder for next steps ---
print("Model loading section complete.")

# --- Rest of API code will go here ---
# --- Define Input Data Model using Pydantic ---
# Features MUST match the columns in X_train_fe (after encoding and FE)
class TransactionFeatures(BaseModel):
    step: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    # One-hot encoded 'type' features (check exact names from L1)
    # Assuming 'type_CASH_IN' was dropped by drop_first=True
    type_CASH_OUT: int = Field(..., description="1 if transaction type is CASH_OUT, else 0", ge=0, le=1)
    type_DEBIT: int = Field(..., description="1 if transaction type is DEBIT, else 0", ge=0, le=1)
    type_PAYMENT: int = Field(..., description="1 if transaction type is PAYMENT, else 0", ge=0, le=1)
    type_TRANSFER: int = Field(..., description="1 if transaction type is TRANSFER, else 0", ge=0, le=1)
    # Feature engineered in L2.1
    amt_ratio_orig: float = Field(..., description="Ratio of amount to original balance")


    class Config:
         schema_extra = {
             "example": {
                 "step": 10, "amount": 5000.0, "oldbalanceOrg": 20000.0, "newbalanceOrig": 15000.0,
                 "oldbalanceDest": 1000.0, "newbalanceDest": 6000.0,
                 "type_CASH_OUT": 1, "type_DEBIT": 0, "type_PAYMENT": 0, "type_TRANSFER": 0,
                 "amt_ratio_orig": 0.25 # 5000 / 20000
             }
         }
print("Pydantic input model defined.")

# --- FastAPI App and Endpoints will go here ---
# --- Create FastAPI App Instance ---
app = FastAPI(
    title="Local Fraud Detection API",
    description="API for predicting fraudulent transactions using XGBoost pipeline (Local Run).",
    version="0.1.0-local"
)
print("FastAPI app instance created.")


# --- Define API Endpoints ---
@app.get("/", tags=["Status"])
async def read_root():
    """Root endpoint providing API status."""
    return {"status": "local_ok", "pipeline_loaded": (pipeline is not None)}


@app.post("/predict", tags=["Prediction"])
async def predict_fraud(features: TransactionFeatures):
    """Receives transaction features and returns a fraud prediction."""
    if pipeline is None:
        print("Prediction attempt failed: Pipeline not loaded.")
        raise HTTPException(status_code=503, detail="Model pipeline is not available.")


    try:
        # 1. Convert Pydantic model to dictionary
        feature_dict = features.dict()


        # 2. Create Pandas DataFrame (single row, ensure column order MATCHES training)
        # Use the expected_features_in list determined during model loading
        if expected_features_in is None or len(expected_features_in) == 0:
             raise HTTPException(status_code=500, detail="Model feature expectations not loaded.")


        try:
             input_df = pd.DataFrame([feature_dict])[expected_features_in] # Ensure order
        except KeyError as e:
             raise HTTPException(status_code=400, detail=f"Missing expected feature in input: {e}")


        # 3. Make prediction
        prediction = pipeline.predict(input_df)
        prediction_value = int(prediction[0])


        # 4. Get probability (optional but recommended)
        probability_fraud = 0.0
        try: # Use try-except as predict_proba might not always be available
            if hasattr(pipeline, "predict_proba"):
                probability = pipeline.predict_proba(input_df)
                probability_fraud = float(probability[0][1]) # Probability of class 1 (Fraud)
            else:
                print("Warning: Pipeline does not support predict_proba.")
        except Exception as proba_e:
            print(f"Warning: Could not get probability: {proba_e}")


        print(f"Prediction made: Label={prediction_value}, Probability={probability_fraud:.4f}")
        return {
            "prediction_label": "Fraud" if prediction_value == 1 else "Not Fraud",
            "prediction_value": prediction_value,
            "is_fraud": bool(prediction_value),
            "probability_fraud": f"{probability_fraud:.4f}"
        }


    except Exception as e:
        print(f"ERROR during prediction request processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error during prediction: {e}")


print("API endpoints defined.")

# --- Local Run Block will go here ---
# --- Section to run locally using uvicorn ---
if __name__ == "__main__":
    print("Running FastAPI app locally using uvicorn...")
    # Note: host='0.0.0.0' makes it accessible on your network
    # Use host='127.0.0.1' to restrict access only to your local machine
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)