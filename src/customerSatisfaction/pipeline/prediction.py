import joblib
import numpy as np
import pandas as pd
import mlflow.sklearn
from pathlib import Path
from customerSatisfaction.config.configuration import ConfigurationManager
from customerSatisfaction import logger

class PredictionPipeline:
    def __init__(self):
        # 1. Access configuration for fallbacks
        config_manager = ConfigurationManager()
        
        # 2. Connect to the Model Registry
        # We pull 'Production' which now contains the BUNDLED (Transformer + Model) pipeline
        try:
            model_uri = "models:/Customer_Satisfaction_Model/Production"
            self.model = mlflow.sklearn.load_model(model_uri)
            logger.info("Successfully loaded Bundled Production model from DagsHub Registry")
            self.using_registry = True
        except Exception as e:
            logger.error(f"Failed to load model from Registry: {e}")
            self.using_registry = False
            # Fallback to local model if registry is unreachable
            model_training_config = config_manager.get_model_training_config()
            self.model = joblib.load(model_training_config.model_path)
            # If using local fallback, we still need the local transformer
            self.transformer = joblib.load(Path('artifacts/feature_transformation/transformer.pkl'))

    def predict(self, data):
        """
        Args:
            data: A pandas DataFrame containing RAW features (before transformation)
        """
        try:
            if self.using_registry:
                # The MLflow model is a Pipeline; it transforms and predicts in one go
                prediction = self.model.predict(data)
            else:
                # Local fallback logic (where model and transformer are separate)
                transformed_data = self.transformer.transform(data)
                prediction = self.model.predict(transformed_data)
            
            return prediction
            
        except Exception as e:
            logger.exception("Prediction failed")
            raise e