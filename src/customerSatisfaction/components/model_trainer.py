import pandas as pd
import joblib
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from customerSatisfaction import logger
from customerSatisfaction.entity.config_entity import ModelTrainingConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        # Mapping string names to actual Sklearn/XGB classes
        self.model_map = {
            "KNN": KNeighborsClassifier,
            "LogisticRegression": LogisticRegression,
            "RandomForest": RandomForestClassifier,
            "XGBoost": xgb.XGBClassifier,
            "MLP": MLPClassifier
        }

    def train(self):
        try:
            # Task 1: Load RAW Training Data (now contains strings/categories)
            train_df = pd.read_csv(self.config.train_data_path)
            X_train_raw = train_df.drop(columns=["target"])
            y_train = train_df["target"]
            
            # --- THE SENIOR FIX: Just-in-Time Transformation ---
            # We load the transformer saved in Stage 4 to turn strings into numbers
            transformer_path = "artifacts/feature_transformation/transformer.pkl" # Adjust if in config
            transformer = joblib.load(transformer_path)
            
            logger.info("Transforming raw data into numeric features for training...")
            X_train_transformed = transformer.transform(X_train_raw)
            logger.info(f"Transformation complete. Numeric features: {X_train_transformed.shape[1]}")

            # 'all_params' structure: { "KNN": {...}, "XGBoost": {...} }
            model_configs = self.config.all_params 

            for model_name, params in model_configs.items():
                logger.info(f"--- Starting Training for: {model_name} ---")
                
                if model_name in self.model_map:
                    model_class = self.model_map[model_name]
                    model_instance = model_class(**params)
                    
                    # Task 3: Training Fit (Using TRANSFORMED numeric data)
                    logger.info(f"Fitting {model_name}...")
                    model_instance.fit(X_train_transformed, y_train)

                    # Task 4: Save Model Artifacts
                    save_path = os.path.join(
                        os.path.dirname(self.config.model_path), 
                        f"{model_name}.joblib"
                    )
                    joblib.dump(model_instance, save_path)
                    logger.info(f"Artifact saved: {save_path}")
                else:
                    logger.warning(f"Model {model_name} not found in model_map. Skipping.")

            logger.info(">>>>>> Stage 5: Multi-Model Training Completed Successfully <<<<<<")

        except Exception as e:
            logger.exception("Multi-Model Training failed")
            raise e