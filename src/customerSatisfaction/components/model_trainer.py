import pandas as pd

import joblib

import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from sklearn.neural_network import MLPClassifier

from catboost import CatBoostClassifier

from customerSatisfaction import logger

from customerSatisfaction.entity.config_entity import ModelTrainingConfig



class ModelTrainer:

    def __init__(self, config: ModelTrainingConfig):

        self.config = config

        

        # Comprehensive model mapping including CatBoost for MLOps pipeline

        self.model_map = {

            "RandomForest": RandomForestClassifier,

            "GradientBoosting": GradientBoostingClassifier,

            "AdaBoost": AdaBoostClassifier,

            "CatBoost": CatBoostClassifier,

            "MLP": MLPClassifier

        }



    def train(self):

        try:

            logger.info("="*80)

            logger.info("STAGE 5: MODEL TRAINING - OPTIMIZING FOR UNSATISFIED CLASS")

            logger.info("="*80)

            

            # 1. LOAD RAW TRAINING DATA

            logger.info("1. Loading Training Data...")

            train_df = pd.read_csv(self.config.train_data_path)

            

            X_train_raw = train_df.drop(columns=[self.config.target_column])

            y_train = train_df[self.config.target_column]

            

            logger.info(f"    [OK] Loaded training data: {train_df.shape}")

            logger.info(f"    [INFO] Class distribution: {y_train.value_counts().to_dict()}")



            # 2. LOAD PREPROCESSOR AND TRANSFORM

            logger.info("2. Loading Preprocessor and Transforming Features...")

            transformer_path = "artifacts/feature_transformation/transformer.pkl"

            

            if not os.path.exists(transformer_path):

                raise FileNotFoundError(f"Preprocessor not found at: {transformer_path}")

            

            transformer = joblib.load(transformer_path)

            X_train_transformed = transformer.transform(X_train_raw)

            logger.info(f"    [OK] Features transformed: {X_train_transformed.shape}")



            # 3. GET MODEL CONFIGURATIONS FROM PARAMS.YAML

            model_configs = self.config.all_params

            

            # 4. TRAIN MODELS

            print("\n" + "="*80)

            print("PIPELINE TRAINING PROGRESS (BALANCED WEIGHTS ENABLED)")

            print("="*80)

            

            trained_count = 0

            

            for model_name, params in model_configs.items():

                if model_name not in self.model_map:

                    logger.warning(f"Skipping {model_name}: Not in model_map")

                    continue

                

                try:

                    print(f"Training: {model_name}...")

                    model_class = self.model_map[model_name]

                    

                    # Log parameters (picks up 'balanced' weights & 20 epochs for MLP)

                    logger.info(f"Initializing {model_name} with: {params}")

                    model_instance = model_class(**params)

                    

                    # Train model

                    # The 'balanced' params in YAML handle the focus on unsatisfied customers here

                    model_instance.fit(X_train_transformed, y_train)

                    

                    # Save individual model artifact

                    # We save them separately so the Evaluation stage can loop through them

                    save_path = os.path.join(

                        os.path.dirname(self.config.model_path), 

                        f"{model_name}.joblib"

                    )

                    joblib.dump(model_instance, save_path)

                    

                    trained_count += 1

                    logger.info(f"    [OK] {model_name} trained and saved to {save_path}")

                    

                except Exception as e:

                    logger.error(f"    [ERROR] Failed to train {model_name}: {str(e)}")

                    print(f"[X] {model_name} training failed.")



            # 5. SUMMARY

            print("="*80)

            print(f"SUCCESSFULLY TRAINED: {trained_count} MODELS")

            print(f"ARTIFACTS FOLDER: {os.path.dirname(self.config.model_path)}")

            print("="*80 + "\n")



            if trained_count == 0:

                raise ValueError("No models were successfully trained!")



        except Exception as e:

            logger.exception("Model Training failed")

            raise e