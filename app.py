import dagshub
import uvicorn
import pandas as pd
import time  # Added for high-resolution timing
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from customerSatisfaction.pipeline.prediction import PredictionPipeline
from customerSatisfaction import logger
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram  # Added Histogram for Alarms

# --- 1. INITIALIZATION ---
try:
    dagshub.init(
        repo_owner='Onabanjomicheal', 
        repo_name='Customer_Satisfaction_Prediction_to_Production', 
        mlflow=True
    )
except Exception as e:
    logger.error(f"DagsHub failed: {e}")

predictor = PredictionPipeline()

# --- ALARM SYSTEM DATA POINTS ---
PREDICTION_COUNT = Counter("predictions_total", "Predictions by result", ["result"])

# This allows an alarm to trigger if P95 latency > 500ms
MODEL_LATENCY = Histogram(
    "model_prediction_duration_seconds",
    "Inference latency distribution",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# This allows an alarm to trigger if error rates spike
PREDICTION_ERRORS = Counter("prediction_errors_total", "Total count of failed inferences")

# --- 2. SCHEMA ---
class CustomerData(BaseModel):
    carrier_handling_time: float
    delivery_time_days: float
    order_items_count: float
    payment_value: float
    estimated_delivery_days: float
    avg_item_price: float
    product_photos_qty: float
    is_weekend_order: int
    order_hour: int
    product_description_lenght: float 
    total_freight: float
    total_price: float
    is_late_delivery: int
    used_installments: float
    payment_installments: float
    order_month: int
    order_day_of_week: int
    product_category_name: str
    payment_type: str
    customer_state: str

# --- 3. APP & MONITORING ---
app = FastAPI(title="Customer Satisfaction Intelligence API")
Instrumentator().instrument(app).expose(app)

@app.post("/predict")
async def predict_route(data: CustomerData):
    try:
        # Start timer for Latency Alarm
        start_time = time.perf_counter()
        
        df = pd.DataFrame([data.model_dump()])
        # Extract probability for Class 1 (Satisfied)
        probs = predictor.model.predict_proba(df)[0]
        sat_prob = float(probs[1])
        
        # Record Latency
        inference_time = time.perf_counter() - start_time
        MODEL_LATENCY.observe(inference_time)
        
        # --- SCENARIO MAPPING ---
        # Scenario A: Operational Action
        if sat_prob >= 0.80:
            interpretation = "Highly Satisfied"
            action = "Automated Follow-up (Thank You)"
            status_code = "GREEN"
        elif 0.60 <= sat_prob < 0.80:
            interpretation = "Satisfied"
            action = "Standard Processing"
            status_code = "GREEN"
        elif 0.40 <= sat_prob < 0.60:
            interpretation = "Neutral / Needs Attention"
            action = "Proactive Support Email"
            status_code = "YELLOW"
        else:
            interpretation = "High Risk"
            action = "Urgent Manager Intervention"
            status_code = "RED"

        # Scenario B: Strategic Risk Level
        if sat_prob >= 0.70: risk_level = "Negligible"
        elif sat_prob >= 0.60: risk_level = "Low"
        elif sat_prob >= 0.40: risk_level = "Moderate"
        else: risk_level = "Critical"

        PREDICTION_COUNT.labels(result=interpretation).inc()
        
        return {
            "status": "success",
            "metadata": {
                "interpretation": interpretation,
                "recommended_action": action,
                "risk_level": risk_level,
                "alert_color": status_code,
                "latency_s": round(inference_time, 4)
            },
            "scores": {
                "satisfaction_probability": round(sat_prob, 4),
                "churn_probability": round(1 - sat_prob, 4)
            }
        }
    except Exception as e:
        # Increment error counter for the Error Alarm
        PREDICTION_ERRORS.inc()
        logger.error(f"Inference Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)