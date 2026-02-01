from zenml.steps import step
from pathlib import Path
from customerSatisfaction.config.configuration import ConfigurationManager
from customerSatisfaction.components.feature_engineering import FeatureEngineering
from customerSatisfaction.utils.common import logger


@step
def feature_engineering_step(validation_status: bool) -> str:
    """
    ZenML Step: Feature Engineering

    Args:
        validation_status (bool): Output from Stage 2

    Returns:
        str: Path to feature-engineered dataset
    """

    if not validation_status:
        raise RuntimeError("Data validation failed. Stopping pipeline.")

    logger.info("Running ZenML Step: Feature Engineering")

    # Load configuration
    config_manager = ConfigurationManager()
    fe_config = config_manager.get_feature_engineering_config()

    validated_dir = Path(fe_config.validated_data_dir)
    output_file = Path(fe_config.feature_engineered_file)

    if not validated_dir.exists():
        raise FileNotFoundError(f"Validated data directory not found: {validated_dir}")

    # Run feature engineering
    fe_component = FeatureEngineering(config=fe_config)
    fe_component.run()

    if not output_file.exists():
        raise FileNotFoundError("Feature engineering output file was not created")

    logger.info(f"Feature engineering completed. Output: {output_file}")

    return str(output_file)
