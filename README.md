# Real-Time Fraud Detection API & Demo (Project 1)

## Project Overview

This project implements a real-time fraud detection system, featuring a machine learning model exposed via a serverless API and demonstrated with an interactive web UI. The primary goal was to build an end-to-end solution leveraging cloud services with a strict zero-cost objective using AWS Free Tier services.

The system can predict whether a given financial transaction is fraudulent based on its features. This project demonstrates skills in data preprocessing, model training, hyperparameter tuning, API development with FastAPI, serverless deployment using AWS Lambda & API Gateway via SAM, and building a user-friendly demo with Streamlit.

**Live Demo (Streamlit Cloud):** [PENDING - URL to be added once Level 4 is complete]
**API Endpoint (AWS Lambda):** https://ino023h7ib.execute-api.ap-southeast-5.amazonaws.com/ _(Note: This is a public demo endpoint)_

---

## Table of Contents

- [Project Goal](#project-goal)
- [Architecture](#architecture)
- [Technologies Used](#technologies-used)
- [Features](#features)
- [Setup & Installation](#setup--installation)
  - [Prerequisites](#prerequisites)
  - [Local ML Model Development](#local-ml-model-development)
  - [Local API Testing](#local-api-testing)
- [AWS Deployment (SAM)](#aws-deployment-sam)
  - [Key AWS Services Used](#key-aws-services-used)
  - [Deployment Steps](#deployment-steps)
- [Demo UI](#demo-ui)
  - [Temporary EC2 Demo (Learning Exercise)](#temporary-ec2-demo-learning-exercise)
  - [Permanent Streamlit Cloud Demo](#permanent-streamlit-cloud-demo)
- [Challenges & Learnings](#challenges--learnings)
- [Future Enhancements](#future-enhancements)
- [Contact](#contact)

---

## Project Goal

To develop and deploy a machine learning model capable of detecting fraudulent transactions in near real-time. The project aims to:
1.  Build a robust fraud detection model using XGBoost.
2.  Perform feature engineering and hyperparameter tuning to optimize model performance (focus on Recall).
3.  Expose the model as a scalable, serverless API using FastAPI on AWS Lambda.
4.  Create an interactive Streamlit application to demonstrate the API's functionality.
5.  Strictly adhere to a zero-cost deployment strategy leveraging AWS Free Tier services.

---

## Architecture

The system comprises the following key components:

1.  **Local Machine Learning Pipeline:**
    *   Data preprocessing, feature engineering, and class imbalance handling (SMOTE).
    *   XGBoost model training and hyperparameter tuning (using Scikit-learn `RandomizedSearchCV`).
    *   Model artifact (`best_fraud_pipeline.joblib`) generation.
2.  **Serverless API Backend (AWS):**
    *   **FastAPI Application (`api.py`):** Defines the prediction endpoint.
    *   **AWS Lambda:** Hosts the FastAPI application using Mangum adapter. Deployed as a container image from ECR. Loads the model from S3 on initialization.
    *   **Amazon API Gateway (HTTP API):** Provides a public HTTPS endpoint to trigger the Lambda function.
    *   **AWS S3:** Stores the trained model artifact (`.joblib` file) to overcome Lambda deployment package size limits.
    *   **Amazon ECR:** Stores the Lambda function's container image.
    *   **AWS SAM (Serverless Application Model):** Used to define and deploy the serverless infrastructure (Lambda, API Gateway, IAM Roles).
    *   **AWS CloudFormation:** Underlies SAM for infrastructure provisioning.
    *   **AWS IAM:** Manages permissions for AWS services.
3.  **Demo User Interface:**
    *   **Streamlit Application (`demo.py`):** Provides a user interface to input transaction features.
    *   Calls the live AWS Lambda API endpoint to get predictions.
    *   Hosted permanently on Streamlit Community Cloud (Planned for Level 4).
    *   (A temporary EC2 deployment was also planned as a learning exercise for VM-based deployments).

_(Consider adding an architecture diagram here later by creating an image, uploading it to your GitHub repo in an `assets` or `images` folder, and using markdown: `![Architecture Diagram](assets/architecture.png)` )_

---

## Technologies Used

*   **Programming Language:** Python 3.11
*   **Machine Learning:**
    *   Pandas, NumPy (Data Manipulation)
    *   Scikit-learn (Pipeline, `train_test_split`, `RandomizedSearchCV`, Metrics)
    *   XGBoost (Classification Model)
    *   Imbalanced-learn (SMOTE for class imbalance)
    *   Joblib (Model saving/loading)
    *   SHAP (Model Explainability - local demo part)
*   **API Development:**
    *   FastAPI (Web framework)
    *   Pydantic (Data validation)
    *   Mangum (Adapter for AWS Lambda)
*   **AWS Cloud Services (Free Tier Focused):**
    *   AWS Lambda
    *   Amazon API Gateway (HTTP API)
    *   Amazon S3 (for model storage & SAM artifacts)
    *   Amazon ECR (for Lambda container image)
    *   AWS SAM (Serverless Application Model) & SAM CLI
    *   AWS CloudFormation
    *   AWS IAM
    *   AWS CloudWatch (for logging and monitoring)
*   **Containerization:** Docker
*   **Demo UI:** Streamlit (Planned for Level 4)
*   **Version Control:** Git & GitHub

---

## Features

*   **Fraud Prediction:** Predicts if a transaction is fraudulent (0 or 1) and provides a probability score.
*   **Serverless API:** Scalable, cost-effective API endpoint hosted on AWS Lambda.
*   **S3 Model Loading:** Efficiently handles large model files by loading from S3.
*   **Containerized Lambda:** Deploys Lambda function as a container image, managing larger dependencies.
*   **Interactive Demo (Planned):** User-friendly Streamlit web application for easy testing and demonstration.
*   **Local Development & Tuning:** Complete workflow for local model development and hyperparameter optimization.
*   **Zero-Cost Deployment (Targeted):** Designed to operate within AWS Free Tier limits (with awareness of potential minor ECR storage costs if free tier is exceeded).

---

## Setup & Installation

*(This section is for others who might want to run your code locally. Ensure your `requirements.txt` in the root of the project is accurate for these local ML steps).*

### Prerequisites

*   Python 3.11
*   Git
*   AWS Account (with AWS CLI configured for `amirul-dev` or equivalent user)
*   AWS SAM CLI (latest version recommended)
*   Docker Desktop (latest version recommended, running)
*   Access to an S3 bucket for model storage (e.g., `aws-sam-cli-managed-default-samclisourcebucket-3ojbej2lkkdk`)

### Local ML Model Development (Project 1 - Levels 1 & 2)

1.  Clone the repository:
    ```bash
    git clone https://github.com/amirulhazym/realtime-fraud-detection-api.git
    cd realtime-fraud-detection-api
    ```
2.  Create and activate a Python virtual environment (e.g., `p1env`):
    ```bash
    python -m venv p1env
    .\p1env\Scripts\activate  # Windows
    # source p1env/bin/activate # Linux/macOS
    ```
3.  Install dependencies (ensure `requirements.txt` in the root primarily lists ML development dependencies):
    ```bash
    pip install -r requirements.txt
    pip install huggingface_hub # If loading data directly via hf://
    ```
4.  Run the Jupyter Notebook `01-Data-Exploration.ipynb` (or your equivalent script for ML model development) to:
    *   Load and sample data.
    *   Perform EDA.
    *   Preprocess data, perform feature engineering.
    *   Split data, apply SMOTE.
    *   Tune hyperparameters using `RandomizedSearchCV`.
    *   Evaluate the tuned model and save `best_fraud_pipeline.joblib` to the project root.

### Local API Testing (Project 1 - Level 2.5)

1.  Ensure `best_fraud_pipeline.joblib` is in the project root.
2.  Create/ensure `api.py` (configured for local Uvicorn, not S3 loading) exists in the project root.
3.  Install FastAPI specific dependencies if not in main `requirements.txt`:
    ```bash
    pip install fastapi "uvicorn[standard]" pydantic
    ```
4.  From the project root, run (ensure `api.py` has the `if __name__ == "__main__": uvicorn.run(...)` block active):
    ```bash
    python api.py
    ```
5.  Access `http://127.0.0.1:8000/docs` in your browser.

---

## AWS Deployment (SAM)

This project utilizes AWS SAM for deploying the API as a containerized AWS Lambda function with an API Gateway trigger. The model is loaded from S3.

### Key AWS Services Used
*   AWS Lambda, Amazon API Gateway (HTTP API), Amazon S3, Amazon ECR, AWS SAM CLI, AWS CloudFormation, AWS IAM.

### Deployment Steps (Manual ECR Push Workflow Summary)

*(Refer to `G-v5.7-Go` document for detailed sub-steps - contact me for the document)*

1.  **Prepare ECR Repository:** Create a private ECR repository (e.g., `fraud-api-repo`) in `ap-southeast-5`.
2.  **Build Local Docker Image:** Using `fraud_api_lambda/Dockerfile` and code.
    ```bash
    docker build -t fraud-api-local:latest -f fraud_api_lambda/Dockerfile fraud_api_lambda/
    ```
3.  **Log in to ECR, Tag, and Push Image:**
    ```bash
    aws ecr get-login-password --region ap-southeast-5 | docker login --username AWS --password-stdin 345594585491.dkr.ecr.ap-southeast-5.amazonaws.com
    docker tag fraud-api-local:latest 345594585491.dkr.ecr.ap-southeast-5.amazonaws.com/fraud-api-repo:latest
    docker push 345594585491.dkr.ecr.ap-southeast-5.amazonaws.com/fraud-api-repo:latest
    ```
4.  **Upload Model (`.joblib`) to S3:** Upload `best_fraud_pipeline.joblib` to an S3 bucket (e.g., `s3://aws-sam-cli-managed-default-samclisourcebucket-3ojbej2lkkdk/models/best_fraud_pipeline.joblib`).
5.  **Configure `fraud_api_lambda/api.py`:** Update to load the model from the S3 path.
6.  **Configure `fraud_api_lambda/requirements.txt`:** Ensure `boto3` and other inference dependencies are listed.
7.  **Modify `template.yaml`:**
    *   Set `PackageType: Image`.
    *   Set `ImageUri` under `Properties` to the ECR image URI (e.g., `345594585491.dkr.ecr.ap-southeast-5.amazonaws.com/fraud-api-repo@sha256:<YOUR_IMAGE_DIGEST>`).
    *   Remove/comment out SAM's `Metadata` block for Docker building.
    *   Ensure `S3ReadPolicy` points to the correct S3 bucket.
8.  **Deploy with SAM:**
    *   Delete any failed CloudFormation stack (`fraud-detection-api-stack`).
    *   Delete `samconfig.toml`.
    *   Run (using the explicit parameters if `--guided` causes issues):
    ```bash
    sam deploy --stack-name fraud-detection-api-stack --region ap-southeast-5 --capabilities CAPABILITY_IAM --resolve-s3 --image-repository 345594585491.dkr.ecr.ap-southeast-5.amazonaws.com/fraud-api-repo
    ```
9.  The API will be accessible at the `FraudApiEndpoint` URL output by SAM (e.g., `https://ino023h7ib.execute-api.ap-southeast-5.amazonaws.com/`).

---

## Demo UI

*(This section will be updated upon completion of Project 1, Level 4)*

### Temporary EC2 Demo (Learning Exercise)
A planned temporary deployment of a Streamlit demo to an AWS EC2 `t2.micro` instance for learning full-stack VM deployment, including local SHAP analysis. This instance will be strictly terminated after testing.

### Permanent Streamlit Cloud Demo
The final user-facing demo will be deployed on Streamlit Community Cloud for portfolio showcasing.
*   **URL:** [PENDING - To be added upon completion of P1 L4]
*   The Streamlit app will call the live AWS Lambda API endpoint for predictions.

---

## Challenges & Learnings

*   **AWS Lambda Deployment Limits:** Encountered Lambda's 250MB unzipped package size limit. Resolved by shifting to S3 model loading and subsequently deploying the Lambda function as a container image from ECR.
*   **AWS SAM CLI & Container Images:** Faced persistent issues with SAM CLI's automated image build-to-deploy linkage on Windows. Successfully worked around this by implementing a manual Docker build and ECR push workflow, then directing SAM to use the pre-pushed `ImageUri`.
*   **Network Connectivity for Docker (Proxied Environment):** `docker build` initially failed with TLS handshake timeouts due to a proxied smartphone hotspot connection. Resolved by using PDAnet (via USB) to establish a more stable network path for Docker operations.
*   **API Runtime Dependencies:** Debugged `Runtime.ImportModuleError` in Lambda (e.g., for `uvicorn`) by ensuring only essential runtime dependencies were included in the Lambda package and that local development tools were not imported in the Lambda version of `api.py`.

---

## Future Enhancements

*   **Advanced Monitoring:** Implement deeper CloudWatch monitoring and alarms for the Lambda and API Gateway.
*   **Security Hardening:** Refine Lambda IAM permissions to Principle of Least Privilege. Explore API Gateway authentication.
*   **CI/CD Integration:** Automate the ECR push and SAM deployment via GitHub Actions (as per Project 2 goals).
*   **Ray Tune Implementation:** Finalize and integrate hyperparameter tuning using Ray Tune's Core API for more advanced optimization.

---

## Contact

Created by amirulhazym - feel free to contact me!
*   [github.com/amirulhazym](https://github.com/amirulhazym)
*   [linkedin.com/in/amirulhazym](https://linkedin.com/in/amirulhazym)
*   [instagram.com/amirulhazym](https://instagram.com/amirulhazym)
*   [amirulhazym@gmail.com](mailto:amirulhazym@gmail.com)

---
