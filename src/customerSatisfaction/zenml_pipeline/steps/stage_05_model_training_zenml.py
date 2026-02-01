from zenml import step
from pathlib import Path
from customerSatisfaction.config.configuration import ConfigurationManager
from customerSatisfaction.components.model_trainer import ModelTrainer
from customerSatisfaction.utils.common import logger


@step(enable_cache=False)
def model_training_step() -> dict:
    """
    ZenML Step: Model Training
    --------------------------
    - Trains regression models
    - Logs experiments to MLflow
    - Saves best model artifact
    - Returns training summary
    """

    logger.info("ZenML Step: Model Training started")

    # 1️⃣ Load configuration
    config_manager = ConfigurationManager()
    model_training_config = config_manager.get_model_training_config()

    # 2️⃣ Run training component
    trainer = ModelTrainer(config=model_training_config)
    training_summary = trainer.train()

    # 3️⃣ Validate model artifact
    model_path = Path(training_summary["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    logger.info(
        f"Model training completed successfully. "
        f"Best model: {training_summary['best_model']}"
    )

    return training_summary
