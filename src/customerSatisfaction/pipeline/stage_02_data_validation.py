from customerSatisfaction.config.configuration import ConfigurationManager
from customerSatisfaction.components.data_validation import DataValidation
from customerSatisfaction import logger

STAGE_NAME = "Data Validation Stage"

class DataValidationPipeline:
    """Pipeline wrapper for Stage 02: Data Validation"""

    def main(self):
        """
        Executes the Data Validation stage:
        1. Fetch configuration
        2. Instantiate DataValidation component
        3. Run validation for all CSVs
        """
        config_manager = ConfigurationManager()
        data_validation_config = config_manager.get_data_validation_config()

        data_validator = DataValidation(config=data_validation_config)
        validated_files = data_validator.run_validation()

        logger.info(f"Validated files: {validated_files}")


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        DataValidationPipeline().main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
