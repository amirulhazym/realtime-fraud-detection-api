# Real-Time Fraud Detection via a Serverless API with AWS and XGBoost (Project 1)

## Project Overview

This project implements an end-to-end real-time fraud detection system, featuring a machine learning model (XGBoost) exposed via a scalable serverless API on AWS, and demonstrated with an interactive web UI hosted on Streamlit Community Cloud. The primary goal was to build a production-mimicking solution while adhering to a strict zero-cost objective by leveraging AWS Free Tier services.

The system predicts whether a given financial transaction is fraudulent based on its features, providing both a classification and a probability score. This project showcases a comprehensive MLOps workflow: data preprocessing, feature engineering, model training (with SMOTE for imbalance), hyperparameter tuning (`RandomizedSearchCV`), API development (FastAPI), serverless deployment (AWS Lambda, API Gateway, SAM, ECR, S3), and building a user-friendly demonstration interface (Streamlit).

**🚀 Live Demo (Streamlit Community Cloud):** 
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://realtime-fraud-detection-api.streamlit.app/) 
**Access the live interactive demo here:** [https://realtime-fraud-detection-api.streamlit.app/](https://realtime-fraud-detection-api.streamlit.app/)

**⚙️ API Endpoint (AWS Lambda):** [https://ino023h7ib.execute-api.ap-southeast-5.amazonaws.com/predict](https://ino023h7ib.execute-api.ap-southeast-5.amazonaws.com/predict) 
_(Note: This is a public demo endpoint with Free Tier limits)_

---

## Table of Contents

- [Project Goal](#project-goal)
- [System Architecture](#system-architecture)
- [Technologies Used](#technologies-used)
- [Core Features](#core-features)
- [Local Development & Setup](#local-development--setup)
  - [Prerequisites](#prerequisites-1)
  - [ML Model Development Workflow](#ml-model-development-workflow)
  - [Local API Testing](#local-api-testing)
- [AWS Serverless API Deployment](#aws-serverless-api-deployment)
  - [Key AWS Services Leveraged](#key-aws-services-leveraged)
  - [Deployment Workflow Highlights](#deployment-workflow-highlights)
- [Interactive Demonstration UIs](#interactive-demonstration-uis) <!-- Updated Title -->
  - [MVP1: EC2-Hosted Full Demo (Development & Learning Showcase)](#ec2-hosted-full-demo-development--learning-showcase) <!-- Updated Title -->
  - [MVP2: Streamlit Community Cloud (Live Portfolio Showcase)](#streamlit-community-cloud-live-portfolio-showcase) <!-- Updated Title -->
- [Key Challenges & Learnings](#key-challenges--learnings)
- [Potential Future Enhancements](#potential-future-enhancements)
- [Contact](#contact)

---

## Project Goal

To develop and deploy a highly performant machine learning model capable of detecting fraudulent financial transactions in near real-time as part of an end-to-end system. The project aimed to:
1.  Build a robust fraud detection model (XGBoost) achieving strong recall for the minority fraud class.
2.  Implement effective feature engineering and hyperparameter tuning (`RandomizedSearchCV`) for optimal model performance.
3.  Expose the trained model as a scalable, resilient, and serverless API using FastAPI on AWS Lambda.
4.  Create two interactive Streamlit applications:
    *   A full-featured version (including SHAP explainability) deployed temporarily on AWS EC2 for comprehensive demonstration and VM deployment practice.
    *   A lightweight, API-driven version deployed permanently on Streamlit Community Cloud for portfolio showcasing.
5.  Strictly adhere to a zero-cost deployment strategy for persistent components, leveraging AWS Free Tier services.

---

## System Architecture

The system integrates several components from local development to cloud deployment, forming an end-to-end solution:

1.  **Machine Learning Pipeline (Local Development):**
    *   Data loading (from Hugging Face Hub), exploration, preprocessing, and feature engineering (e.g., `amt_ratio_orig`).
    *   Handling class imbalance using SMOTE.
    *   Training an XGBoost classification model within a Scikit-learn pipeline.
    *   Hyperparameter optimization using `RandomizedSearchCV`.
    *   Serialization of the final model artifact (`best_fraud_pipeline.joblib`).
2.  **Serverless API Backend (AWS):**
    *   **FastAPI Application (`fraud_api_lambda/api.py`):** Defines the `/predict` endpoint, handles request validation (Pydantic), and performs inference.
    *   **AWS S3:** Securely stores the trained model artifact (`best_fraud_pipeline.joblib`), loaded by Lambda at initialization to bypass deployment size limits.
    *   **AWS Lambda:** Hosts the FastAPI application (via Mangum adapter). Deployed as a Docker container image.
    *   **Amazon ECR (Elastic Container Registry):** Stores the Lambda function's Docker container image.
    *   **Amazon API Gateway (HTTP API):** Provides a public, secure HTTPS endpoint that triggers the Lambda function.
    *   **AWS SAM (Serverless Application Model) & SAM CLI:** Used to define (`template.yaml`), build, and deploy the entire serverless backend infrastructure.
    *   **AWS CloudFormation:** Works under the hood of SAM to provision and manage AWS resources as code.
    *   **AWS IAM:** Manages granular permissions for all AWS services, ensuring secure interactions.
3.  **Interactive User Interfaces (Streamlit):**
    *   **EC2 Demo (`demo.py`):** A full-featured Streamlit application temporarily deployed on an AWS EC2 `t3.micro` instance. This version loads the model locally on EC2 to provide SHAP-based model explainability alongside API predictions. *This instance was terminated after testing to manage costs.*
    *   **Streamlit Community Cloud Demo (`streamlit_app_cloud/streamlit_cloud_demo.py`):** A lightweight, API-driven Streamlit application hosted permanently and free of charge. This version calls the live AWS Lambda API for predictions and serves as the primary public demo.

---

## Technologies Used

*   **Programming Language:** Python 3.11
*   **Data Science & Machine Learning:**
    *   Pandas, NumPy (Data Manipulation & Numerical Operations)
    *   Scikit-learn (Pipeline, `train_test_split`, `RandomizedSearchCV`, Metrics)
    *   XGBoost (Core Classification Model)
    *   Imbalanced-learn (SMOTE for Oversampling)
    *   Joblib (Model Serialization/Deserialization)
    *   SHAP (Lundberg & Lee - Model Explainability, used in EC2 demo)
*   **API Development:**
    *   FastAPI (High-performance Web Framework)
    *   Pydantic (Data Validation & Settings Management)
    *   Mangum (Adapter for running ASGI applications like FastAPI on AWS Lambda)
*   **AWS Cloud Services (Primarily Free Tier):**
    *   AWS Lambda (Serverless Compute)
    *   Amazon API Gateway (HTTP API for Serverless Invocation)
    *   Amazon S3 (Object Storage for Model Artifacts & SAM templates)
    *   Amazon ECR (Container Registry for Lambda Images)
    *   AWS SAM (Serverless Application Model) & SAM CLI (Deployment Framework)
    *   AWS CloudFormation (Infrastructure as Code)
    *   AWS IAM (Identity & Access Management)
    *   AWS EC2 (Elastic Compute Cloud - for temporary VM-based demo)
    *   AWS CloudWatch (Logging & Basic Monitoring)
*   **Containerization:** Docker
*   **UI Development:** Streamlit
*   **Version Control & CI/CD (Foundation for Project 2):** Git & GitHub

---

## Core Features

*   **Real-Time Fraud Prediction:** Classifies transactions as fraudulent (1) or legitimate (0) and provides a granular probability score.
*   **Serverless & Scalable API:** Robust API endpoint built for scalability and cost-efficiency using AWS Lambda and API Gateway.
*   **Efficient Model Handling:** Utilizes S3 for model storage and ECR for containerized Lambda deployment to manage dependencies and model size effectively.
*   **Interactive Demonstrations:**
    *   **Live Cloud Demo:** A publicly accessible Streamlit app showcasing API interaction.
    *   **Full EC2 Demo Learnings:** Demonstrated deployment to a VM including local model loading for SHAP explanations.
*   **Model Explainability (EC2 Demo):** Integrated SHAP to provide insights into feature contributions for individual predictions.
*   **Optimized End-to-End ML Workflow:** Includes data preprocessing, feature engineering, SMOTE for imbalance, and `RandomizedSearchCV` for hyperparameter tuning.
*   **Zero-Cost Goal for Persistent Components:** Designed to operate within AWS Free Tier limits for the Lambda API and Streamlit Community Cloud hosting.

---

## Local Development & Setup

*(This section guides replication of local development stages).*

### Prerequisites <!-- Note: Changed this to Prerequisites-1 for ToC link if "Setup & Installation" has sub-headers -->

*   Python 3.11
*   Git
*   AWS Account (with AWS CLI v2 configured for an IAM user with necessary permissions)
*   AWS SAM CLI (latest version)
*   Docker Desktop (running)

### ML Model Development Workflow (Project 1 - Levels 1 & 2)

1.  Clone the repository:
    ```bash
    git clone https://github.com/amirulhazym/realtime-fraud-detection-api.git
    cd realtime-fraud-detection-api
    ```
2.  Create and activate a Python virtual environment (e.g., `p1env` with Python 3.11):
    ```bash
    python3.11 -m venv p1env
    source p1env/bin/activate # Linux/macOS
    # .\p1env\Scripts\activate # Windows
    ```
3.  Install core ML dependencies (refer to the main `requirements.txt` in the project root):
    ```bash
    pip install -r requirements.txt 
    ```
4.  Execute Jupyter Notebooks (e.g., `01-Data-Exploration.ipynb`, etc.) or corresponding Python scripts to:
    *   Load data (e.g., from Hugging Face Hub: `datasets.load_dataset("fraud-detection-bank")['train'].to_pandas().sample(100000, random_state=42)`)
    *   Perform Exploratory Data Analysis (EDA).
    *   Preprocess data (scaling, encoding) and engineer features.
    *   Split data into training and testing sets.
    *   Apply SMOTE to the training set to handle class imbalance.
    *   Define a Scikit-learn pipeline with preprocessing steps and an XGBoost classifier.
    *   Tune hyperparameters using `RandomizedSearchCV` focusing on optimizing `recall`.
    *   Evaluate the tuned model on the test set.
    *   Save the best pipeline as `best_fraud_pipeline.joblib` in the project root.

### Local API Testing (Project 1 - Level 2.5)

1.  Ensure `best_fraud_pipeline.joblib` is in the project root.
2.  The `api.py` in the project root should be configured for local Uvicorn testing (i.e., loads model locally, not from S3, and includes the `if __name__ == "__main__": uvicorn.run(...)` block).
3.  Install FastAPI specific dependencies (if not in main `requirements.txt`):
    ```bash
    pip install fastapi "uvicorn[standard]" pydantic
    ```
4.  From the project root, run:
    ```bash
    python api.py
    ```
5.  Access API documentation and test at `http://127.0.0.1:8000/docs`.

---

## AWS Serverless API Deployment

The core API is deployed using AWS SAM, featuring a containerized AWS Lambda function loading its model from S3 and triggered by API Gateway, forming a key part of the end-to-end solution.

### Key AWS Services Leveraged
*   AWS Lambda, Amazon API Gateway (HTTP API), Amazon S3, Amazon ECR, AWS SAM CLI, AWS CloudFormation, AWS IAM.

### Deployment Workflow Highlights (Manual ECR Push & S3 Model)

*(Detailed sub-steps are part of the G-v5.7-Go internal project plan)*

1.  **S3 Model Upload:** `best_fraud_pipeline.joblib` is uploaded to a designated S3 bucket (e.g., `s3://<your-sam-bucket>/models/best_fraud_pipeline.joblib`).
2.  **Lambda Code (`fraud_api_lambda/api.py`):** Modified to download and load the model from this S3 path into the Lambda's `/tmp` directory during initialization. `boto3` added to `fraud_api_lambda/requirements.txt`.
3.  **ECR Image Push (Manual Workflow):**
    *   A private ECR repository (e.g., `fraud-api-repo`) is created in the target region (e.g., `ap-southeast-5`).
    *   A Docker image is built locally using `fraud_api_lambda/Dockerfile`.
    *   The image is tagged and pushed to the ECR repository.
4.  **SAM Template (`template.yaml`):**
    *   Configured with `PackageType: Image`.
    *   `ImageUri` property points directly to the pushed ECR image digest/tag.
    *   Includes an IAM policy (`S3ReadPolicy`) granting the Lambda function permission to read from the S3 model bucket.
    *   SAM's internal Docker build metadata is removed.
5.  **SAM Deployment:**
    *   `sam deploy --guided` (or with explicit parameters) is used to deploy the CloudFormation stack, creating the Lambda function (from ECR image), API Gateway, and necessary IAM roles.
6.  The live API becomes accessible via the `FraudApiEndpoint` URL provided by the SAM deployment output.

---

## Interactive Demonstration UIs <!-- Updated Title -->

Two Streamlit UIs were developed and deployed to demonstrate the system:

### MVP1: EC2-Hosted Full Demo (Development & Learning Showcase) <!-- Updated Title -->

*   **Purpose:** To practice full-stack deployment to a traditional VM (AWS EC2 `t3.micro`) and showcase model explainability using SHAP, which requires loading the model file locally within the Streamlit application's environment. This formed a complete end-to-end test from UI to model.
*   **Script:** `demo.py` (in the project root).
*   **Deployment Steps:**
    1.  Launched a new EC2 `t3.micro` instance (Amazon Linux 2023) with increased storage (30GiB).
    2.  Installed Git, Python 3.11, and `pip`.
    3.  Cloned the GitHub repository onto the EC2 instance.
    4.  Created and activated a Python 3.11 virtual environment.
    5.  Installed all dependencies from `ec2_requirements.txt` (after resolving version and temporary disk space issues by redirecting `TMPDIR`).
    6.  Securely copied (`scp`) `best_fraud_pipeline.joblib` from local machine to the EC2 instance.
    7.  Ran the Streamlit application: `streamlit run demo.py --server.port 8501 --server.address 0.0.0.0 &`.
    8.  Accessed and tested the full demo (API calls + SHAP plots) via the EC2 public IP.
*   **Status:** **This EC2 instance might be TERMINATED** later to adhere cost-management best practices and the project's zero-cost focus for ongoing hosted components.

### MVP2: Streamlit Community Cloud (Live Portfolio Showcase) <!-- Updated Title -->

*   **Purpose:** To provide a persistent, publicly accessible, and zero-AWS-hosting-cost demonstration of the live fraud detection API. This version focuses on the API interaction and serves as the primary shareable deliverable for this end-to-end project.
*   **Script:** `streamlit_app_cloud/streamlit_cloud_demo.py` (refactored from `demo.py` to remove local model loading and SHAP dependencies).
*   **Deployment:**
    1.  Created a dedicated subdirectory `streamlit_app_cloud/`.
    2.  Placed `streamlit_cloud_demo.py` and a minimal `requirements.txt` (specific to this cloud app) within this subdirectory.
    3.  Pushed changes to GitHub.
    4.  Deployed from GitHub to Streamlit Community Cloud, specifying `streamlit_app_cloud/streamlit_cloud_demo.py` as the main application file and ensuring Python 3.11 was selected for the environment.
*   **Live URL:** [https://realtime-fraud-detection-api.streamlit.app/](https://realtime-fraud-detection-api.streamlit.app/)

---

## Key Challenges & Learnings

This project provided significant hands-on experience and several learning opportunities in building an end-to-end ML system:

*   **AWS Lambda Deployment Constraints:** Overcame Lambda's 250MB unzipped package size limit by strategically moving to S3 for model storage and then further to containerized Lambda deployment via ECR for managing larger dependencies and ensuring environment consistency.
*   **AWS SAM CLI & Container Images:** Troubleshot and resolved issues with AWS SAM CLI's automated container image handling on Windows by adopting a more robust manual Docker build and ECR push workflow, then configuring SAM to use the `ImageUri`. This provided deeper insight into the underlying deployment mechanisms.
*   **EC2 Environment Setup & Dependency Management:**
    *   Successfully navigated Python version management on EC2 (installing Python 3.11 alongside system Python and using virtual environments).
    *   Resolved `pip install` failures due to specific package version incompatibilities and temporary disk space (`/tmp` tmpfs) limitations by redirecting `TMPDIR` and adjusting storage.
*   **Network Issues for Docker:** Resolved `docker build` TLS handshake timeouts in a proxied network environment by switching to a direct USB tethered connection (PDAnet).
*   **Git Workflow for Divergent Branches:** Managed divergent Git branches between local and EC2 environments using `git pull --no-rebase` to merge changes before pushing.
*   **Cost Optimization & Resource Management:** Diligently practiced terminating EC2 resources after use (EC2 Full Demo) and leveraging free tiers (Lambda, API Gateway, S3, ECR Free Tier, Streamlit Community Cloud) for persistent components.
*   **Iterative UI/UX Refinement:** Progressed from a basic Streamlit UI to a more polished version based on iterative feedback and design considerations for different deployment targets.

---

## Potential Future Enhancements

*   **CI/CD Pipeline (Project 2):** Fully automate the model retraining, ECR image push, and SAM deployment pipeline using GitHub Actions.
*   **Advanced Monitoring & Alerting:** Implement comprehensive CloudWatch dashboards, metrics, and alarms for the API's performance, error rates, and costs.
*   **Security Hardening:**
    *   Refine Lambda IAM role to strictly adhere to the Principle of Least Privilege.
    *   Implement API Gateway authentication/authorization (e.g., API Keys, IAM, or Lambda Authorizers).
*   **A/B Testing for Models:** Design infrastructure to support A/B testing of different model versions in production.
*   **Batch Prediction Workflow:** Add capability for processing batch predictions from files stored in S3.
*   **Advanced Hyperparameter Tuning:** Fully integrate Ray Tune's Core API for more sophisticated and distributed hyperparameter optimization.
*   **Data Drift & Model Retraining Trigger:** Implement mechanisms to detect data drift and trigger automated model retraining pipelines.

---

## Contact

Created by **Amirulhazym**
*   GitHub: [github.com/amirulhazym](https://github.com/amirulhazym)
*   LinkedIn: [linkedin.com/in/amirulhazym](https://linkedin.com/in/amirulhazym)
*   Instagram: [@amirulhazym](https://instagram.com/amirulhazym)
*   Email: [amirulhazym@gmail.com](mailto:amirulhazym@gmail.com)

---
