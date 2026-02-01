# stage_02_data_validation_zenml.py

from zenml.steps import step
from pathlib import Path
from customerSatisfaction.config.configuration import ConfigurationManager
from customerSatisfaction.components.data_validation import DataValidation
from customerSatisfaction import logger

@step
def data_validation_step(raw_data_path: str) -> bool:
    """
    ZenML step for data validation.
    Validates all datasets in raw_data_path using schema.yaml.
    
    Args:
        raw_data_path (str): Path to raw ingested data from Stage 1.
    
    Returns:
        bool: True if all datasets pass validation, False otherwise.
    """
    logger.info("Running ZenML Step: Data Validation")

    # Load configuration
    config_manager = ConfigurationManager()
    data_validation_config = config_manager.get_data_validation_config()

    # Override unzip_data_dir to use the path from Stage 1
    data_validation_config.unzip_data_dir = Path(raw_data_path)

    # Initialize DataValidation component
    validator = DataValidation(config=data_validation_config)

    # Run validation
    validation_status = validator.validate_all_columns()

    if validation_status:
        logger.info("Data validation successful. All datasets meet schema requirements.")
    else:
        logger.warning(f"Data validation completed with issues. Check {data_validation_config.report_file}")

    return validation_status
