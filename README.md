# Customer Satisfaction Prediction (End-to-End MLOps)

## Project Overview

This project implements an end-to-end Machine Learning pipeline for predicting customer satisfaction, following production-grade MLOps practices.  
The system is modular, configuration-driven, and designed to ensure data quality before any transformation or modeling occurs.

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

## Pipeline Stages

### Stage 01 – Data Ingestion
- Reads raw datasets
- Organizes artifacts into structured directories
- Controlled via configuration files

---

### Stage 02 – Data Validation (Completed)

This stage ensures data quality before downstream processing.

**Validation checks include:**
- Column existence validation
- Data type enforcement
- Numeric min/max bounds
- Allowed categorical values
- Critical columns null checks
- Maximum null threshold per dataset
- Schema-driven validation using `schema.yaml`

**Outputs:**
- Validated datasets:



- Overall validation status file

**Observed result:**
- Most datasets passed validation
- `olist_products_dataset.csv` was correctly flagged due to data quality issues
- Datasets not defined in the schema were safely skipped and logged

---

## Why This Stage Matters

- Prevents invalid data from entering transformation and modeling stages
- Makes data issues observable and auditable
- Enables safe automation for new incoming data

---

## How to Run the Project

### Step 1: Clone the Repository
```bash
git clone https://github.com/Onabanjomicheal/Customer_Satisfaction_Prediction_to_Production
cd Customer_Satisfaction_Prediction_to_Production

conda create -n cnncls python=3.10 -y


python app.py



### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/Onabanjomicheal/Customer_Satisfaction_Prediction_to_Production \
MLFLOW_TRACKING_USERNAME=Onabanjomicheal \
MLFLOW_TRACKING_PASSWORD=f1c60ac6579043df34bebaf39959b928ca4d6fd7 \
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/Onabanjomicheal/Customer_Satisfaction_Prediction_to_Production.mlflow
export MLFLOW_TRACKING_USERNAME=Onabanjomicheal
export MLFLOW_TRACKING_PASSWORD=f1c60ac6579043df34bebaf39959b928ca4d6fd7


import dagshub
dagshub.init(repo_owner='Onabanjomicheal', repo_name='Customer_Satisfaction_Prediction_to_Production', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)