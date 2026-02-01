import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
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
            # 1. Load Data
            test_data = pd.read_csv(self.config.test_data_path)
            X_test_raw = test_data.drop(columns=[self.config.target_column])
            y_test = test_data[self.config.target_column]
            
            # 2. Load the Transformer (to bundle it)
            # Adjust this path based on your config if needed
            transformer = joblib.load('artifacts/feature_transformation/transformer.pkl')
            
            # 3. Get local model files
            model_dir = os.path.dirname(self.config.model_path)
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
            
            performance_tracker = []

            with mlflow.start_run(run_name="Bundled_Comparison_Session") as parent:
                for model_file in model_files:
                    model_name = model_file.replace('.joblib', '')
                    
                    with mlflow.start_run(run_name=model_name, nested=True):
                        logger.info(f"Evaluating: {model_name}")
                        
                        # Load the raw model weights
                        model_weights = joblib.load(os.path.join(model_dir, model_file))
                        
                        # --- THE SENIOR FIX: Bundle into a Pipeline ---
                        full_pipeline = Pipeline([
                            ("preprocessor", transformer),
                            ("classifier", model_weights)
                        ])
                        
                        # Predict using the RAW data (Pipeline handles transformation)
                        y_pred = full_pipeline.predict(X_test_raw)
                        
                        # Metrics
                        metrics = {
                            "accuracy": accuracy_score(y_test, y_pred),
                            "f1_macro": f1_score(y_test, y_pred, average='macro', zero_division=0),
                            "precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
                            "recall": recall_score(y_test, y_pred, average='macro', zero_division=0)
                        }
                        
                        mlflow.log_metrics(metrics)
                        
                        # Log Signature based on RAW input
                        signature = infer_signature(X_test_raw, y_pred)
                        
                        # Log the ENTIRE pipeline as the model artifact
                        mlflow.sklearn.log_model(
                            sk_model=full_pipeline, 
                            artifact_path="model",
                            signature=signature,
                            input_example=X_test_raw.iloc[[0]]
                        )
                        
                        performance_tracker.append({"name": model_name, **metrics})

                # 4. Promote Champion
                best_model = max(performance_tracker, key=lambda x: x['f1_macro'])
                logger.info(f"Winner: {best_model['name']} (F1: {best_model['f1_macro']:.4f})")

                client = mlflow.tracking.MlflowClient()
                runs = client.search_runs(
                    experiment_ids=[parent.info.experiment_id],
                    filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}' AND tags.mlflow.runName = '{best_model['name']}'"
                )
                
                if runs:
                    self._promote_to_production(runs[0].info.run_id, best_model['name'])

            self._print_leaderboard(performance_tracker)

        except Exception as e:
            logger.exception("Model evaluation failed")
            raise e

    def _promote_to_production(self, run_id, model_name):
        reg_name = "Customer_Satisfaction_Model"
        result = mlflow.register_model(f"runs:/{run_id}/model", reg_name)
        
        # Move to Production stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=reg_name, version=result.version, stage="Production", archive_existing_versions=True
        )
        logger.info(f"Version {result.version} is now in Production.")

    def _print_leaderboard(self, tracker):
        print("\n" + "="*70)
        print(f"{'Model':<20} | {'F1':<8} | {'Acc':<8} | {'Prec':<8} | {'Rec':<8}")
        print("-" * 70)
        for m in tracker:
            print(f"{m['name']:<20} | {m['f1_macro']:.4f} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f}")
        print("="*70 + "\n")