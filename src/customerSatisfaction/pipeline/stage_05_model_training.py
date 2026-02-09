from customerSatisfaction.config.configuration import ConfigurationManager
from customerSatisfaction.components.model_trainer import ModelTrainer
from customerSatisfaction import logger

STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline:
    """
    Stage 05: Model Training
    ----------------------------------------
    - Orchestrates the training of multiple models (KNN, XGBoost, CatBoost, etc.)
    - Injects doctoral research constraints (e.g., 20 epochs for MLP)
    - Saves trained .joblib artifacts for the Stage 06 Tournament
    """

    def __init__(self):
        # We leave __init__ empty so main.py can call it without arguments
        pass

    def main(self):
        try:
            # 1. Initialize Configuration Manager
            config_manager = ConfigurationManager()

            # 2. Get Model Training Configuration
            # This contains paths to train data and the 'all_params' dictionary
            model_training_config = config_manager.get_model_training_config()

            # 3. Initialize the ModelTrainer Component
            # We pass the config here, fulfilling the requirement of the Component
            model_trainer = ModelTrainer(config=model_training_config)

            # 4. Run the training logic
            # This will iterate through your model_map and save the joblib files
            model_trainer.train()

        except Exception as e:
            logger.exception(f"Fatal error during {STAGE_NAME}: {e}")
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e