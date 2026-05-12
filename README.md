# MLOps Forecasting & Monitoring System

## Overview

This project demonstrates an end-to-end MLOps pipeline for demand forecasting using XGBoost, FastAPI, Streamlit, and AWS EC2.

The system goes beyond model training by implementing:

* Real-time prediction serving
* Prediction logging
* Drift detection using KS statistics
* Automated retraining workflow
* Monitoring dashboard
* Cloud deployment on AWS EC2

The goal of the project was to simulate a lightweight production ML monitoring environment instead of building just another notebook-based ML model.

---

# Architecture

```text
User Request
     ↓
FastAPI Prediction API
     ↓
Prediction Logging
     ↓
Drift Detection (KS Test)
     ↓
Automatic Retraining
     ↓
Monitoring Dashboard
     ↓
AWS EC2 Deployment
```

---

# Tech Stack

| Component            | Technology    |
| -------------------- | ------------- |
| Model Training       | XGBoost       |
| Backend API          | FastAPI       |
| Monitoring Dashboard | Streamlit     |
| Drift Detection      | SciPy KS Test |
| Data Processing      | Pandas        |
| Visualization        | Matplotlib    |
| Deployment           | AWS EC2       |
| Model Serialization  | Joblib        |

---

# Features

## Real-Time Prediction API

* Built using FastAPI
* Interactive Swagger UI for testing
* Live prediction generation

## Prediction Logging

* Every prediction request is logged
* Tracks timestamp, actual values, and predictions
* Used for monitoring and drift analysis

## Drift Detection

* Kolmogorov-Smirnov statistical test
* Detects distribution shift between training and production data
* Configurable drift threshold

## Automated Retraining

* Triggered automatically when drift is detected
* Retrains XGBoost model using updated data
* Saves refreshed model artifacts

## Monitoring Dashboard

* Built using Streamlit
* Displays:

  * Total predictions
  * RMSE
  * Drift status
  * Retraining status
  * Prediction trends
  * Actual vs predicted comparison

## Cloud Deployment

* Deployed on AWS EC2
* Publicly accessible API and dashboard
* Lightweight deployment using free-tier infrastructure

---

# Project Structure

```text
mlops-forecasting-monitoring/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── logs/
│   ├── predictions.csv
│   └── retraining_status.txt
│
├── models/
│   └── model.pkl
│
├── src/
│   ├── app.py
│   ├── dashboard.py
│   ├── train.py
│   ├── check_drift.py
│   ├── feature_engineering.py
│   └── prediction_logger.py
│
├── requirements.txt
└── README.md
```

---

# Running Locally

## 1. Clone Repository

```bash
git clone https://github.com/Mahesh-Hemadri/mlops-forecasting-monitoring.git
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

## 3. Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Train Model

```bash
python src/train.py
```

---

# Run FastAPI Server

```bash
uvicorn src.app:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

# Run Dashboard

```bash
streamlit run src/dashboard.py
```

Dashboard URL:

```text
http://localhost:8501
```

---

# Run Drift Detection

```bash
python src/check_drift.py
```

---

# AWS Deployment

The application was deployed on AWS EC2 using:

* Ubuntu EC2 instance
* FastAPI for serving
* Streamlit for monitoring
* Security group configuration for public access

---

# Key Learnings

This project helped strengthen practical understanding of:

* ML system design
* Production monitoring
* Drift detection workflows
* Model retraining strategies
* API deployment
* AWS EC2 deployment
* Real-world debugging and environment management

---

# Future Improvements

* Docker containerization
* CI/CD integration
* Airflow scheduling
* Model versioning
* Advanced observability
* Prometheus + Grafana monitoring
* Cloud storage integration

---

# Screenshots

Check out /assets


---

# Author

Mahesh Hemadri
