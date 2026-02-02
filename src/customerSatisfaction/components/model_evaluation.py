import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from customerSatisfaction.entity.config_entity import ModelEvaluationConfig
from customerSatisfaction import logger
from mlflow.models.signature import infer_signature

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment("Customer_Satisfaction_Evaluation")

    def evaluate(self):
        try:
            # 1. Load RAW Data (Now contains headers like price, payment_type, etc.)
            test_data = pd.read_csv(self.config.test_data_path)
            X_test_raw = test_data.drop(columns=[self.config.target_column])
            y_test = test_data[self.config.target_column]
            
            # 2. Load the Transformer (The "Recipe")
            transformer_path = "artifacts/feature_transformation/transformer.pkl"
            transformer = joblib.load(transformer_path)
            
            # 3. Access local model files
            model_dir = os.path.dirname(self.config.model_path)
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
            
            performance_tracker = []

            # Master run for comparing all model variants
            with mlflow.start_run(run_name="Bundled_Inference_Session") as parent:
                for model_file in model_files:
                    model_name = model_file.replace('.joblib', '')
                    
                    with mlflow.start_run(run_name=model_name, nested=True):
                        logger.info(f"Evaluating Bundled Pipeline for: {model_name}")
                        
                        # Load raw weights
                        model_weights = joblib.load(os.path.join(model_dir, model_file))
                        
                        # --- THE BUNDLE ---
                        # We fuse the Preprocessor and the Classifier here.
                        # This object can handle RAW data directly.
                        full_pipeline = Pipeline([
                            ("preprocessor", transformer),
                            ("classifier", model_weights)
                        ])
                        
                        # Predict using RAW features
                        y_pred = full_pipeline.predict(X_test_raw)
                        
                        # Calculate Metrics
                        metrics = {
                            "accuracy": accuracy_score(y_test, y_pred),
                            "f1_macro": f1_score(y_test, y_pred, average='macro', zero_division=0),
                            "precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
                            "recall": recall_score(y_test, y_pred, average='macro', zero_division=0)
                        }
                        
                        mlflow.log_metrics(metrics)
                        
                        # Log Signature (Schema of the RAW inputs)
                        signature = infer_signature(X_test_raw, y_pred)
                        
                        # Log the ENTIRE pipeline as the production artifact
                        mlflow.sklearn.log_model(
                            sk_model=full_pipeline, 
                            artifact_path="model",
                            signature=signature,
                            input_example=X_test_raw.iloc[[0]]
                        )
                        
                        performance_tracker.append({"name": model_name, **metrics})

                # 4. Identify and Promote the Champion
                best_model = max(performance_tracker, key=lambda x: x['f1_macro'])
                logger.info(f"Champion Selected: {best_model['name']} with F1: {best_model['f1_macro']:.4f}")

                client = mlflow.tracking.MlflowClient()
                # Find the specific run ID for the champion
                runs = client.search_runs(
                    experiment_ids=[parent.info.experiment_id],
                    filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}' AND tags.mlflow.runName = '{best_model['name']}'"
                )
                
                if runs:
                    self._promote_to_production(runs[0].info.run_id, best_model['name'])

            self._print_leaderboard(performance_tracker)

        except Exception as e:
            logger.exception("Final Evaluation failed")
            raise e

    def _promote_to_production(self, run_id, model_name):
        reg_name = "Customer_Satisfaction_Model"
        result = mlflow.register_model(f"runs:/{run_id}/model", reg_name)
        
        # Transition to Production alias/stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=reg_name, version=result.version, stage="Production", archive_existing_versions=True
        )
        logger.info(f"Model version {result.version} successfully transitioned to 'Production'.")

    def _print_leaderboard(self, tracker):
        print("\n" + "═"*70)
        print(f"║ {'Model Name':<20} ║ {'F1 Score':<8} ║ {'Accuracy':<8} ║ {'Recall':<8} ║")
        print("╟" + "─"*68 + "╢")
        for m in tracker:
            print(f"║ {m['name']:<20} ║ {m['f1_macro']:.4f}   ║ {m['accuracy']:.4f}   ║ {m['recall']:.4f}   ║")
        print("═"*70 + "\n")