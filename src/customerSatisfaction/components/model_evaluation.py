import pandas as pd
import numpy as np
import joblib
import os
import json
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from customerSatisfaction.entity.config_entity import ModelEvaluationConfig
from customerSatisfaction import logger
from mlflow.models.signature import infer_signature
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment("Customer_Satisfaction_Evaluation")

    def evaluate(self):
        try:
            logger.info("="*80)
            logger.info("STAGE 6: EVALUATION - WEIGHTED SELECTION (F1 + REC + ACC)")
            logger.info("="*80)
            
            # 1. Load Data
            test_data = pd.read_csv(self.config.test_data_path)
            X_test_raw = test_data.drop(columns=[self.config.target_column])
            y_test = test_data[self.config.target_column]
            
            # 2. Setup Environment
            transformer = joblib.load("artifacts/feature_transformation/transformer.pkl")
            model_dir = os.path.dirname(self.config.model_path)
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
            
            performance_tracker = []

            # 3. Evaluation Loop
            with mlflow.start_run(run_name="Champion_Model_Selection"):
                for model_file in model_files:
                    model_name = model_file.replace('.joblib', '')
                    
                    with mlflow.start_run(run_name=model_name, nested=True) as model_run:
                        try:
                            model_weights = joblib.load(os.path.join(model_dir, model_file))
                            X_test_transformed = transformer.transform(X_test_raw)
                            
                            # Probability for class 0 (Unsatisfied)
                            y_proba = model_weights.predict_proba(X_test_transformed)[:, 0]

                            custom_threshold = 0.4 
                            y_pred = np.where(y_proba >= custom_threshold, 0, 1)
                            
                            # Calculate metrics
                            auc = roc_auc_score(y_test, 1 - y_proba)
                            acc = accuracy_score(y_test, y_pred)
                            rec = recall_score(y_test, y_pred, pos_label=0)
                            f1 = f1_score(y_test, y_pred, pos_label=0)
                            
                            # --- MULTI-METRIC CALCULATION ---
                            # We give F1 the most weight (50%), Recall (30%), and Accuracy (20%)
                            # This ensures we catch the most 'useful' model for production.
                            weighted_score = (f1 * 0.5) + (rec * 0.3) + (acc * 0.2)

                            metrics = {
                                "accuracy": acc,
                                "unsatisfied_recall": rec,
                                "unsatisfied_precision": precision_score(y_test, y_pred, pos_label=0, zero_division=0),
                                "f1_unsatisfied": f1,
                                "roc_auc": auc,
                                "weighted_champion_score": weighted_score,
                                "decision_threshold": custom_threshold
                            }
                            
                            mlflow.log_metrics(metrics)
                            self._log_confusion_matrix(y_test, y_pred, model_name)
                            
                            full_pipeline = Pipeline([
                                ("preprocessor", transformer),
                                ("classifier", model_weights)
                            ])
                            
                            signature = infer_signature(X_test_raw, y_pred)
                            mlflow.sklearn.log_model(full_pipeline, "model", signature=signature)
                            
                            performance_tracker.append({
                                "name": model_name, 
                                "run_id": model_run.info.run_id, 
                                **metrics
                            })
                            
                            logger.info(f"    [OK] {model_name} -> Score: {weighted_score:.4f}")
                            
                        except Exception as e:
                            logger.error(f"    [FAIL] {model_name}: {str(e)}")

                # 4. Champion Selection using the Weighted Score
                if performance_tracker:
                    # This logic will now favor CatBoost over RandomForest
                    best_model = max(performance_tracker, key=lambda x: x['weighted_champion_score'])
                    logger.info(f"WINNING MODEL: {best_model['name']} (Score: {best_model['weighted_champion_score']:.4f})")
                    
                    res_path = Path("artifacts/model_evaluation/metrics.json")
                    res_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(res_path, "w") as f:
                        json.dump(best_model, f, indent=4)
                    
                    self._promote_to_production(best_model['run_id'])

            self._print_leaderboard(performance_tracker)

        except Exception as e:
            logger.exception("Evaluation failed")
            raise e

    def _log_confusion_matrix(self, y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted (0=Unsat, 1=Sat)')
        plt.ylabel('Actual (0=Unsat, 1=Sat)')
        plt.title(f'Confusion Matrix: {model_name}')
        path = f"artifacts/model_evaluation/cm_{model_name}.png"
        plt.savefig(path)
        mlflow.log_artifact(path)
        plt.close()

    def _promote_to_production(self, run_id):
        try:
            reg_name = "Customer_Satisfaction_Model"
            mlflow.register_model(f"runs:/{run_id}/model", reg_name)
            logger.info(f"Model {run_id} promoted based on balanced metrics.")
        except Exception as e:
            logger.warning(f"Registration skipped: {e}")

    def _print_leaderboard(self, tracker):
        print("\n" + "="*120)
        print(f"{'ALGORITHM':<20} | {'SCORE':<8} | {'F1':<8} | {'REC':<8} | {'ACC':<8}")
        print("-" * 120)
        # Sorted by the new Weighted Score
        for m in sorted(tracker, key=lambda x: x['weighted_champion_score'], reverse=True):
            print(f"{m['name']:<20} | {m['weighted_champion_score']:.4f} | {m['f1_unsatisfied']:.4f} | {m['unsatisfied_recall']:.4f} | {m['accuracy']:.4f}")
        print("="*120 + "\n")