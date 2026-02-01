from customerSatisfaction import logger
from customerSatisfaction.config.configuration import ConfigurationManager
from customerSatisfaction.components.data_validation import DataValidation

STAGE_NAME = "Data Validation Stage"

class DataValidationPipeline:
    """
    Pipeline class for orchestrating the Data Validation & Merging stage.
    
    This stage triggers the component that performs:
    1. Loading 2. Deduplication 3. Datetime Parsing 4. Merging 
    5. Imputation 6. Export 7. Reporting
    """

    def __init__(self):
        pass

    def main(self) -> bool:
        """
        Executes the Data Validation and Merging stage flow.
        """
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")

        try:
            # Step 1: Initialize Configuration
            logger.info("Initializing configuration for Data Validation...")
            config_manager = ConfigurationManager()
            data_validation_config = config_manager.get_data_validation_config()
            logger.info("Configuration loaded successfully.")

            # Step 2: Initialize Component
            logger.info("Instantiating the DataValidation component...")
            validator = DataValidation(config=data_validation_config)

            # Step 3: Trigger Core Logic (The 7 Tasks)
            logger.info("Triggering the validation and merge logic...")
            status = validator.initiate_data_validation()

            if status:
                logger.info(
                    f"{STAGE_NAME} successful. Merged dataset created and validated."
                )
            else:
                logger.error(
                    f"{STAGE_NAME} failed. Check the generated status/report files."
                )

            logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
            return status

        except Exception as e:
            logger.exception(f"Fatal error during {STAGE_NAME} orchestration: {e}")
            raise e