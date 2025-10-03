# Serverless Real-Time Fraud Detection API 🛡️

[![AWS Serverless API](https://img.shields.io/badge/AWS-Serverless_API-FF9900?logo=amazonaws)](https://zeir21qzal.execute-api.ap-southeast-5.amazonaws.com/predict)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?logo=streamlit)](https://realtime-fraud-detection-api.streamlit.app/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Project Overview

This project implements an end-to-end real-time fraud detection system, featuring a machine learning model (XGBoost) exposed via a scalable serverless API on AWS. The system is demonstrated with an interactive web UI hosted on Streamlit Community Cloud. The primary goal was to build a production-mimicking solution while adhering to a strict **zero-cost objective** by leveraging, utilizing and maximizing AWS Free Tier services.

The system predicts whether a given financial transaction is fraudulent based on its features, providing both a classification and a probability score. This project covers the full MLOps lifecycle: data preprocessing, feature engineering, model training, robust API development (FastAPI), serverless deployment (AWS Lambda, SAM), and building a user-friendly demonstration interface (Streamlit).

## ✨ Live Demo & API Endpoint

- **🚀 Interactive Web App (Streamlit Cloud):** [**https://realtime-fraud-detection-api.streamlit.app/**](https://realtime-fraud-detection-api.streamlit.app/)
- **⚙️ Live API Endpoint (AWS Lambda):** [**https://zeir21qzal.execute-api.ap-southeast-5.amazonaws.com/predict**](https://zeir21qzal.execute-api.ap-southeast-5.amazonaws.com/predict)
![Web Interface](https://framerusercontent.com/images/jTC3mMkJpOEPOBb11e2MHW5l5qw.png?width=873&height=848)


** **More screenshots are available in my portfolio website: [Visit Website](https://amirulhazym.framer.ai/work/fraud-detection-api)**


## 🏗️ System Architecture

The system integrates local ML development with a cloud-native serverless backend and a web-based frontend.
![Web Interface](https://framerusercontent.com/images/BmjEZFvz9hRuKMZKo22kylubBM.png?width=1625&height=824)


## ⭐ Core Features

- **Real-Time Fraud Prediction:** Classifies transactions as fraudulent or legitimate and provides a probability score.
- **Serverless & Scalable API:** Built with FastAPI and deployed on AWS Lambda for high scalability and cost-efficiency.
- **Containerized Deployment:** Uses Docker and Amazon ECR to manage dependencies and ensure environment consistency, overcoming Lambda's size limits.
- **Efficient Model Handling:** The trained model artifact is stored in S3 and loaded by the Lambda function at initialization.
- **Interactive Demo:** A publicly accessible Streamlit app showcases live API interaction.
- **Model Explainability (EC2 Version):** A temporary, full-featured version was deployed to an EC2 VM to demonstrate SHAP-based feature importance.
- **Zero-Cost for Production:** Designed to operate entirely within the AWS Free Tier and Streamlit Community Cloud's free hosting.

## 📶 Model Performance & Results

The final XGBoost model was evaluated on a held-out, imbalanced test set. Given the high cost of missing a fraudulent transaction, the primary metric for success was **Recall** for the 'Fraud' class, with an initial project target of >= 75%.

Through a rigorous process of data preprocessing (SMOTE for class imbalance) and hyperparameter tuning (`RandomizedSearchCV`), the final model significantly exceeded this target.

### Key Evaluation Metrics (Fraud Class)

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Recall** | **98.12%** | **Primary Goal Exceeded.** The model successfully identified **261 out of 266** actual fraud cases in the test set. Only 5 fraud cases were missed. |
| **Precision** | **76.32%** | When the model flagged a transaction as 'Fraud', it was correct **76% of the time**. This maintains a good balance, minimizing false alarms for legitimate customers. |
| **F1-Score** | **0.8586** | Represents a strong balance between Recall and Precision, indicating a robust and effective model. |

---

### Detailed Classification Report (Tuned Model)

```text
               precision    recall  f1-score   support

Not Fraud (0)     1.0000    1.0000    1.0000    199734
    Fraud (1)     0.7632    0.9812    0.8586       266

     accuracy                         1.0000    200000
    macro avg     0.8816    0.9906    0.9293    200000
 weighted avg     1.0000    1.0000    1.0000    200000
```


### Confusion Matrix
The confusion matrix provides a clear view of the model's predictions versus the actual outcomes.

*   **True Positives (TP): 261** (Correctly identified fraud)
*   **False Negatives (FN): 5** (Missed fraud cases - **THE MOST CRITICAL ERROR**)
*   **False Positives (FP): 81** (Legitimate transactions incorrectly flagged as fraud)
*   **True Negatives (TN): 199,653** (Correctly identified legitimate transactions)

## 🛠️ Technology Stack

| Category | Technologies Used |
|----------|------------------|
| **Data Science & ML** | Python 3.11, Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn (SMOTE), Joblib, SHAP |
| **API Development** | FastAPI, Pydantic, Mangum (for Lambda) |
| **Cloud & Deployment** | AWS (Lambda, API Gateway, S3, ECR, SAM, CloudFormation, IAM), Docker, Streamlit Community Cloud, AWS EC2 (for dev demo) |
| **Version Control** | Git & GitHub |

## 📁 Project Structure

```
realtime-fraud-detection-api/
├── .git/
├── p1env/                     # Python virtual environment (in .gitignore)
├── fraud_api_lambda/          # Source code for the Lambda function
│   ├── api.py                 # FastAPI application script (loads model from S3)
│   ├── Dockerfile             # Dockerfile for the Lambda container image
│   └── requirements.txt       # Python dependencies for the Lambda function
├── streamlit_app_cloud/       # Source for the permanent Streamlit Cloud demo
│   ├── streamlit_cloud_demo.py# The API-only Streamlit app script
│   └── requirements.txt       # Minimal dependencies for the cloud app
├── .dockerignore
├── .gitignore
├── best_fraud_pipeline.joblib # The trained model artifact
├── demo.py                    # Script for the full demo (API + SHAP) on EC2
├── ec2_requirements.txt       # Dependencies for the full EC2 demo
├── requirements.txt           # Core project dependencies for local development
├── template.yaml              # AWS SAM template for serverless infrastructure
└── README.md                  # This file
```

## 🚀 Local Setup & Usage

### Prerequisites

- Git, Python 3.11, Docker Desktop
- An AWS account with the AWS CLI v2 and AWS SAM CLI installed and configured.

### Installation & Local Workflow

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/amirulhazym/realtime-fraud-detection-api.git
   cd realtime-fraud-detection-api
   ```

2. **Set Up Virtual Environment:**
   ```bash
   python -m venv p1env
   .\p1env\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the ML Training Pipeline:** Execute the notebooks or scripts to train the model and generate `best_fraud_pipeline.joblib`.

5. **Test API Locally:** Modify a local version of `api.py` to load the model from the local file path instead of S3, then run `uvicorn api:app --reload`. Test endpoints at http://127.0.0.1:8000/docs.

## 🔧 Deployment Deep Dive

### AWS Serverless API

The API deployment was a key part of this project, involving a manual but robust workflow to handle large dependencies.

1. **Model Storage:** The final `best_fraud_pipeline.joblib` was uploaded to an AWS S3 bucket.
2. **Containerization:** The FastAPI application was containerized using the `fraud_api_lambda/Dockerfile`. This image was manually built and pushed to a private repository in Amazon ECR.
3. **Infrastructure as Code (IaC):** The `template.yaml` file defines all AWS resources. It specifies a Lambda function with `PackageType: Image` and points its `ImageUri` to the pre-pushed image in ECR. It also includes an IAM policy granting the Lambda function permission to read the model file from S3.
4. **Deployment:** The `sam deploy` command was used to execute the CloudFormation stack, provisioning the Lambda function, API Gateway, and all necessary IAM roles.

### Streamlit UIs

Two UIs were created to serve different purposes:

1. **EC2 Full Demo (Learning Exercise):** A full-featured version with SHAP explainability was deployed to an EC2 t3.micro instance. This taught valuable lessons in VM environment setup, dependency management on Linux, and troubleshooting. Currently, this instance being monitor actively to ensure zero ongoing cost, might terminate at anytime.
2. **Streamlit Cloud Demo (Live Portfolio):** A lightweight, API-only version was created and deployed to the free Streamlit Community Cloud. This serves as the permanent, public-facing demo for the project.

## 💡 Key Challenges & Learnings

### **1. Challenge: Advanced Hyperparameter Tuning (Ray Tune)**
- **The Problem:** In Phase 2, the initial plan was to use Ray Tune. However, running it locally produced a ModuleNotFoundError because the tune-sklearn library it depends on is deprecated and incompatible with modern Ray versions. An attempt on Colab resulted in a ValueError related to data serialization.

- **The Solution:** Pivoted to a more stable, industry-standard alternative: Scikit-learn's `RandomizedSearchCV`. This allowed for effective hyperparameter tuning without being blocked by tooling issues, ensuring the project could proceed with a well-optimized model.

### **2. Challenge: Overcoming AWS Lambda's Deployment Package Size Limit**
- **The Problem:** Initial attempts to deploy the FastAPI application failed because the total size of the deployment package—which included all code, dependencies, and the `best_fraud_pipeline.joblib` model file—exceeded AWS Lambda's 250 MB unzipped size limit. The deployment would result in an "Internal Server Error" and `CREATE_FAILED` status in CloudWatch logs.

- **The Solution:** The strategy was pivoted to separate the model artifact from the code. The large `.joblib` model file was uploaded to Amazon S3. The Lambda function's code was then modified to use the `boto3` library to download and load the model directly from S3 at runtime, which kept the deployment package size within the acceptable limits.

### **3. Challenge: Unexpected Issues with AWS SAM's Automated Image Deployment**
- **The Problem:** The AWS Serverless Application Model (SAM) CLI's internal logic for automatically building, pushing, and deploying a Docker image to Amazon ECR failed. The `sam deploy --guided` command would report "No images found to deploy," even though the Docker image was built locally. This bug prevented a streamlined, single-command deployment.

- **The Solution:** An alternative, more robust, and manual approach was adopted. The Docker image was manually built and tagged locally using `docker build` and `docker tag`, then manually pushed to the ECR repository using `docker push`. The `template.yaml` file was updated to explicitly reference the image's URI in ECR, and the `sam deploy` command was run without the `--guided` flag, effectively bypassing the buggy part of the SAM CLI. This approach gave more control and ensured a successful deployment.

### **4. Challenge: Managing Costs and Ensuring Zero-Cost Operation**
- **The Problem:** A core mandate for the project was to incur absolutely zero cost by strictly adhering to the **AWS Free Tier**. Using services like Amazon EC2 for the full Streamlit demo posed a high risk of accidental charges if instances were not terminated promptly.

- **The Solution:** A two-part strategy was implemented. First, **AWS Budgets** were set up with mandatory alerts for any spend over $0.01 to act as a safety net. Second, for the long-term demo, the Streamlit app was refactored and deployed to **Streamlit Community Cloud**, which provides a permanent, free hosting solution that doesn't rely on AWS Free Tier limits. The EC2 instance was used only as a temporary learning step and currently being monitor for its usage.

### **5. Challenge: Correctly Installing Python Packages on the EC2 Instance**
- **The Problem:** When deploying the Streamlit demo to an EC2 instance, the `pip install` command failed with "ERROR: Could not find a version that satisfies the requirement...". This was caused by a mismatch between the package versions in the local Python 3.11 development environment and the EC2 instance's default Python 3.9 environment.
  
- **The Solution:** The solution was to explicitly install Python 3.11 on the EC2 instance and create a virtual environment with it, ensuring the Python version on the deployment target matched the development environment. This allowed the project's exact package versions to be installed successfully.

### **6. Challenge: SSH Connection & Permissions on EC2**
- **The Problem:** In Phase 4 (MVP 1), the initial SSH connection to the EC2 instance failed with Permission denied (publickey) due to the .pem key file having "permissions are too open" on the local Windows machine. The connection also dropped frequently.
  
- **The Solution:** Corrected Windows file permissions (ACLs) on the .pem key file by removing access for broad groups like "Authenticated Users." The connection dropping was stabilized by using the ServerAliveInterval option in the SSH command.

### **7. Challenge: The Final API 500 Internal Server Error After a Long Break**
- **The Problem:** After 3 months of project completion, the live API endpoint, which was previously functional, began returning a generic `500 Internal Server Error`. Critically, when investigating the issue, there were **no new logs being generated in AWS CloudWatch** for any of the failed requests. This **"silent failure"** indicated that the problem was not a bug in my Python code, but a more fundamental infrastructure failure occurring before the Lambda function could even start its execution environment.

- **The Solution:** The diagnostic process involved a systematic check of the Lambda function's core dependencies:
    - **CloudWatch Analysis:** The absence of new logs ruled out application-level errors and            pointed towards a failure in the Lambda service's invocation or initialization phase.
    - **ECR Repository Check:** I hypothesized that the container image, the function's primary
      dependency, might be missing. A check in the Amazon Elastic Container Registry (ECR)
      confirmed this: the private ECR repository (fraud-api-repo) that the Lambda function was
      configured to pull from had been deleted, likely during a prior manual resource cleanup.
    - **Path to Resolution:** The initial fix was to recreate the ECR repository and push the image
      again. This led to the next, even deeper, challenge.

### **8. Challenge: Debugging a Cascade of Failures with ECR Public Repositories in AWS SAM**
- **The Problem:** To fix `500 Internal Server Error`, a strategic decision was made to switch to an ECR Public repository to leverage its more generous **"Always Free"** tier. However, this seemingly simple change led to persistent deployment failures. CloudFormation repeatedly failed with a cryptic **"Source image ... is not valid"** error, even after using `docker buildx` to guarantee a correct `x86_64` image architecture. Furthermore, the SAM CLI's transform engine failed with `Invalid Serverless Application Specification` errors when trying to define IAM policies for the function.
  
- **The Solution:** The root cause was diagnosed as an underlying incompatibility or series of bugs in how the AWS SAM/CloudFormation toolchain handles Lambda functions sourced from ECR Public repositories. After exhausting all targeted fixes, the most effective engineering decision was to pivot the strategy back to using **Private ECR repository**. The final template.yaml was reconfigured for the private repo, which immediately resolved all deployment errors and allowed the API to go live successfully. This was a critical lesson in recognizing tooling limitations and prioritizing stable, standard architectures.

## 🔮 Future Enhancements

- **CI/CD Pipeline:** Fully automate the model retraining, ECR image push, and SAM deployment using GitHub Actions (implemented in [Project 2](https://github.com/amirulhazym/mlops-automated-pipeline)).
- **Advanced Monitoring:** Implement comprehensive CloudWatch dashboards and alarms for API performance, errors, and costs.
- **Security Hardening:** Refine the Lambda's IAM role to the principle of least privilege and add an API key to the API Gateway.
- **Database Integration:** Replace the simple API with a more robust system that interacts with a database (Amazon RDS or DynamoDB) to log all prediction requests and their outcomes. This would enable data versioning and ongoing model monitoring.
- **Extended Feature Set:** Expand the project to handle additional data types, such as location data, device information, or historical user spending patterns, to create a more comprehensive fraud detection model.
- **Cost & Performance Optimization:** Fine-tune the Lambda function's memory and CPU allocation for better performance, and explore using a single endpoint for both model inference and explainability to reduce latency and resource usage.
- **Refactor for Explainable AI (XAI) on Streamlit Cloud:** Implement a solution to run SHAP on the Streamlit Community Cloud without exceeding free tier limitations, perhaps by sending a request to the API with an `explain` flag or by using a less resource-intensive explainability method.

## 👤 Author

**Amirulhazym**

- LinkedIn: [linkedin.com/in/amirulhazym](https://linkedin.com/in/amirulhazym)
- GitHub: [github.com/amirulhazym](https://github.com/amirulhazym)
- Portfolio: [amirulhazym.framer.ai](https://amirulhazym.framer.ai)
