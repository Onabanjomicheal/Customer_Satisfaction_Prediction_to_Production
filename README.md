# 🛡️ Customer Satisfaction Prediction (End-to-End MLOps)

## 🚀 Project Overview
This project implements a production-grade, end-to-end Machine Learning pipeline designed to predict customer satisfaction in an e-commerce environment. The system moves beyond simple binary classification to provide **Decision Intelligence**, categorizing customers into actionable risk tiers based on model confidence.

The pipeline is modular, configuration-driven, and ensures strict data quality enforcement before any modeling occurs.

The pipeline emphasizes:
- Reproducibility
- Data quality enforcement
- Clear separation of concerns
- Production-oriented structure

---

## Workflow Architecture

The project follows a clean, scalable workflow:

1. Update `config.yaml`
2. Update `secrets.yaml` (Optional)
3. Update `params.yaml`
4. Update entities
5. Update the configuration manager
6. Update components
7. Update pipeline stages
8. Update `main.py`
9. Update `dvc.yaml`
10. Update `app.py`

---

## 🏗️ Workflow Architecture
The project follows a scalable, professional workflow:
1.  **Config Management:** Update `config.yaml`, `params.yaml`, and `schema.yaml`.
2.  **Infrastructure:** Define entities and the Configuration Manager.
3.  **Components:** Develop modular Python blocks for Ingestion, Validation, and Transformation.
4.  **Orchestration:** Pipeline stages triggered via `main.py` and tracked via `dvc.yaml`.
5.  **Deployment:** FastAPI production backend and Streamlit decision portal.

---

## 📊 Pipeline Stages
### Stage 01 – Data Ingestion
- Reads raw datasets
- Organizes artifacts into structured directories
- Controlled via configuration files

---

### **Stage 02 – Data Validation**
Ensures data quality before downstream processing. Validation checks include:
- **Column Existence:** Ensures all required features are present.
- **Data Type Enforcement:** Matches incoming data types to the schema.
- **Numeric Bounds:** Validates min/max values for financial and logistics data.
- **Quality Gates:** Enforces maximum null thresholds per dataset.
- *Result:* `olist_products_dataset.csv` was correctly flagged due to quality issues, preventing "garbage-in" modeling.

### **Stage 07 – Inference & Decision Logic**
The system transforms raw model probabilities into **Strategic Business Tiers**:
* **70% - 100% (Satisfied):** Green Status — Automated Approval/Standard Processing.
* **30% - 70% (Neutral/At-Risk):** Yellow Status — Triggers Proactive Customer Outreach.
* **< 30% (Dissatisfied):** Red Status — Urgent Manager Intervention Required.

---

## 🛠️ Tech Stack & Monitoring
| Layer | Technology |
| :--- | :--- |
| **Model** | CatBoost (Optimized via MLflow) |
| **Registry** | DagsHub / MLflow |
| **Backend API** | FastAPI / Pydantic |
| **Frontend UI** | Streamlit |
| **Monitoring** | Prometheus / Prometheus-FastAPI-Instrumentator |

---

## 📈 Monitoring & Metrics
The system integrates **Prometheus** to track real-time performance.
- **Metrics Endpoint:** Access `/metrics` for system health and prediction counts.
- **Counter:** Tracks `predictions_total` labeled by result (Satisfied, Neutral, Dissatisfied).

---

## 🏃 How to Run

### **1. Setup Environment**
### Step 1: Clone the Repository
```bash
git clone https://github.com/Onabanjomicheal/Customer_Satisfaction_Prediction_to_Production
cd Customer_Satisfaction_Prediction_to_Production

conda create -n cnncls python=3.10 -y


python app.py



### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/
MLFLOW_TRACKING_USERNAME=
MLFLOW_TRACKING_PASSWORD=
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/
export MLFLOW_TRACKING_USERNAME=
export MLFLOW_TRACKING_PASSWORD=
export DAGSHUB_TOKEN=


import dagshub
dagshub.init(repo_owner='Onabanjomicheal', repo_name='Customer_Satisfaction_Prediction_to_Production', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)


