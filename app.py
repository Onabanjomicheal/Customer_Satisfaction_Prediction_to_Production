from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import pandas as pd
import dagshub
import os
from customerSatisfaction.pipeline.prediction import PredictionPipeline
from customerSatisfaction import logger

# --- NEW: Monitoring Imports ---
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge

# 1. Global Initialization
try:
    dagshub.init(
        repo_owner='Onabanjomicheal', 
        repo_name='Customer_Satisfaction_Prediction_to_Production', 
        mlflow=True
    )
    logger.info("DagsHub initialized successfully.")
except Exception as e:
    logger.error(f"DagsHub initialization failed: {e}")

predictor = PredictionPipeline()

# --- NEW: Define ML-Specific Metric ---
# This Gauge tracks the last prediction: 1 for Satisfied, 0 for Dissatisfied
PREDICTION_SCORE = Gauge("model_prediction_value", "Last predicted satisfaction score (1=Satisfied, 0=Dissatisfied)")

# 2. Pydantic Schema
class CustomerData(BaseModel):
    price: float
    freight_value: float
    product_category_name: str
    product_name_lenght: float
    product_description_lenght: float
    product_photos_qty: float
    product_weight_g: float
    product_length_cm: float
    product_height_cm: float
    product_width_cm: float
    seller_state: str
    customer_state: str
    payment_sequential: float
    payment_type: str
    payment_installments: float
    payment_value: float
    review_availability: int
    purchase_delivery_difference: float
    estimated_actual_delivery_difference: float
    price_category: str
    purchase_delivery_diff_per_price: float

app = FastAPI(
    title="Customer Satisfaction Inference Service",
    description="Real-time inference using the Production champion from DagsHub"
)

# --- NEW: Initialize & Expose Metrics ---
# This creates the /metrics endpoint and tracks latency automatically
instrumentator = Instrumentator().instrument(app)

@app.on_event("startup")
async def _startup():
    instrumentator.expose(app)

@app.get("/")
async def root():
    return {
        "status": "API is running", 
        "model_source": "DagsHub Model Registry",
        "endpoint": "/predict",
        "metrics": "/metrics"
    }

@app.post("/predict")
async def predict_route(data: CustomerData):
    try:
        df = pd.DataFrame([data.model_dump()])
        prediction = predictor.predict(df)
        
        # Convert prediction to integer (0 or 1)
        pred_value = int(prediction[0])
        
        # --- NEW: Update ML Metric ---
        PREDICTION_SCORE.set(pred_value)
        
        result = "Satisfied" if pred_value == 1 else "Dissatisfied"
        
        logger.info(f"Prediction successful: {result}")
        return {
            "prediction": result,
            "status_code": 200
        }
    
    except Exception as e:
        logger.error(f"Prediction route failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)