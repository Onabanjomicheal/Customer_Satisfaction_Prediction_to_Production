import joblib
import numpy as np
import pandas as pd
import mlflow.sklearn
from pathlib import Path
from customerSatisfaction.config.configuration import ConfigurationManager
from customerSatisfaction import logger

class PredictionPipeline:
    def __init__(self):
        config_manager = ConfigurationManager()
        
        try:
            # 1. Connect to the Model Registry (DagsHub)
            model_uri = "models:/Customer_Satisfaction_Model/Production"
            self.model = mlflow.sklearn.load_model(model_uri)
            logger.info("Successfully loaded Bundled Production model from DagsHub Registry")
            self.using_registry = True
        except Exception as e:
            logger.error(f"Failed to load model from Registry: {e}")
            self.using_registry = False
            
            # 2. Local Fallback
            model_training_config = config_manager.get_model_training_config()
            # Note: Ensure model_path points to the specific champion (e.g., CatBoost.joblib)
            self.model = joblib.load(model_training_config.model_path)
            self.transformer = joblib.load(Path('artifacts/feature_transformation/transformer.pkl'))

    def predict(self, data: pd.DataFrame):
        """
        Args:
            data: Raw features from FastAPI request
        """
        try:
            # STANDARD CLEANUP: Drop non-feature IDs that cause conversion errors
            drop_cols = ["order_id", "customer_id", "product_id", "seller_id"]
            data_cleaned = data.drop(columns=[c for c in drop_cols if c in data.columns])

            # CUSTOM THRESHOLD: Using 0.4 as established in Stage 6 Handshake
            threshold = 0.4

            if self.using_registry:
                # self.model is a Pipeline (Transformer + Classifier)
                # We use predict_proba to apply our specific 0.4 threshold
                probabilities = self.model.predict_proba(data_cleaned)[:, 0]
            else:
                # Local fallback transformation
                transformed_data = self.transformer.transform(data_cleaned)
                probabilities = self.model.predict_proba(transformed_data)[:, 0]
            
            # Class 0 = Unsatisfied, Class 1 = Satisfied
            prediction = np.where(probabilities >= threshold, 0, 1)
            
            return prediction
            
        except Exception as e:
            logger.exception("Prediction failed")
            raise e