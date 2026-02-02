import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from sklearn.metrics import classification_report, f1_score
from customerSatisfaction.entity.config_entity import ModelEvaluationConfig
from customerSatisfaction import logger
from mlflow.models.signature import infer_signature

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        # DagsHub integration uses this to point tracking to their servers
        mlflow.set_tracking_uri(self.config.mlflow_uri)

    def evaluate(self):
        try:
            # 1. Load Test Data
            test_data = pd.read_csv(self.config.test_data_path)
            X_test = test_data.drop(columns=[self.config.target_column])
            y_test = test_data[self.config.target_column]
            
            # 2. Identify models in local artifact dir
            model_dir = os.path.dirname(self.config.model_path)
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
            
            leaderboard = {}
            best_model_info = {"model": None, "name": None, "f1": -1.0, "run_id": None}

            # Parent Run: Groups all model attempts in DagsHub UI
            with mlflow.start_run(run_name="Model_Tournament_Pipeline"):
                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                
                for model_file in model_files:
                    model_name = model_file.replace('.joblib', '')
                    
                    # Nested Run: Individual model performance
                    with mlflow.start_run(run_name=model_name, nested=True) as child_run:
                        logger.info(f"Evaluating: {model_name}")
                        
                        model = joblib.load(os.path.join(model_dir, model_file))
                        y_pred = model.predict(X_test)
                        
                        # 3. Calculate Metrics
                        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
                        
                        mlflow.log_metrics({
                            "accuracy": report['accuracy'],
                            "macro_f1": f1_macro,
                            "c0_recall": report.get('0', {}).get('recall', 0)
                        })
                        
                        # ADJUSTED: MLflow 3.x uses 'name' to identify the LoggedModel.
                        # This creates the artifact folder '/model' inside the run.
                        signature = infer_signature(X_test, y_pred)
                        mlflow.sklearn.log_model(
                            sk_model=model, 
                            name="model", 
                            signature=signature,
                            input_example=X_test.iloc[[0]]
                        )
                        
                        leaderboard[model_name] = {"f1": round(f1_macro, 4)}

                        # Tournament Logic: Keep track of the winner
                        if f1_macro > best_model_info["f1"]:
                            best_model_info.update({
                                "model": model, "name": model_name, 
                                "f1": f1_macro, "run_id": child_run.info.run_id
                            })

                # 4. REGISTRATION: Only if tracking is remote (DagsHub)
                if best_model_info["run_id"] and tracking_url_type_store != "file":
                    self._register_champion(best_model_info)

                self._print_results(leaderboard)

        except Exception as e:
            logger.exception("Evaluation stage failed")
            raise e

    def _register_champion(self, best_info):
        """Register winner to DagsHub Model Registry and promote to Production."""
        try:
            reg_name = "Customer_Satisfaction_Model"
            # URI points to the 'model' artifact from the winning run
            model_uri = f"runs:/{best_info['run_id']}/model"
            
            logger.info(f"Registering Champion: {best_info['name']}")
            
            # Register version
            result = mlflow.register_model(model_uri, reg_name)
            
            # Promote to Production for FastAPI deployment
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=reg_name, 
                version=result.version, 
                stage="Production", 
                archive_existing_versions=True
            )
            logger.info(f"SUCCESS: {best_info['name']} promoted to PRODUCTION.")
            
        except Exception as e:
            logger.error(f"Registry Handshake Failed: {e}")

    def _print_results(self, leaderboard):
        print("\n" + "="*40 + "\nTournament Results:\n" + "="*40)
        for name, m in leaderboard.items():
            print(f"{name:<20} | F1: {m['f1']}")
        print("="*40 + "\n")