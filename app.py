from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import pandas as pd
import dagshub
import os
from customerSatisfaction.pipeline.prediction import PredictionPipeline
from customerSatisfaction import logger

# 1. Global Initialization
# Initialize DagsHub once at startup
try:
    dagshub.init(
        repo_owner='Onabanjomicheal', 
        repo_name='Customer_Satisfaction_Prediction_to_Production', 
        mlflow=True
    )
    logger.info("DagsHub initialized successfully.")
except Exception as e:
    logger.error(f"DagsHub initialization failed: {e}")

# Load the model into memory ONCE when the server starts
# This makes individual predictions lightning fast
predictor = PredictionPipeline()

# 2. Pydantic Schema (Matches your Model Signature)
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

@app.get("/")
async def root():
    return {
        "status": "API is running", 
        "model_source": "DagsHub Model Registry",
        "endpoint": "/predict"
    }

@app.post("/predict")
async def predict_route(data: CustomerData):
    try:
        # Convert Pydantic model to DataFrame using model_dump() (Pydantic v2)
        df = pd.DataFrame([data.model_dump()])
        
        # Use the pre-loaded global predictor
        prediction = predictor.predict(df)
        
        # Map prediction to human-readable label
        # (Assuming 1 is Satisfied, 0 is Dissatisfied based on your previous code)
        result = "Satisfied" if int(prediction[0]) == 1 else "Dissatisfied"
        
        logger.info(f"Prediction successful: {result}")
        return {
            "prediction": result,
            "status_code": 200
        }
    
    except Exception as e:
        logger.error(f"Prediction route failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Port 8080 is excellent for AWS App Runner or Elastic Beanstalk
    uvicorn.run(app, host="0.0.0.0", port=8080)