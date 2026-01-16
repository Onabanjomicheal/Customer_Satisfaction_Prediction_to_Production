import logging
from customerSatisfaction.config.configuration import ConfigurationManager
from customerSatisfaction.components.data_validation import DataValidation

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

STAGE_NAME = "Data Validation Stage"

class DataValidationPipeline:
    def __init__(self):
        pass

    def main(self):
        logging.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")

        try:
            # Load configuration (schema embedded in config)
            config_manager = ConfigurationManager()
            data_validation_config = config_manager.get_data_validation_config()

            # Initialize validator with config
            validator = DataValidation(config=data_validation_config)

            # Run validation and corrections
            status = validator.validate_all_columns()

            # Log outcome with path to validated datasets
            if status:
                logging.info(
                    f"{STAGE_NAME} completed successfully. "
                    f"Validated datasets are saved in: {validator.validated_data_dir}"
                )
            else:
                logging.warning(
                    f"{STAGE_NAME} completed with issues. "
                    f"Check validation report and validated datasets in: {validator.validated_data_dir}"
                )

            logging.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
            return status

        except Exception as e:
            logging.error(f"Error in {STAGE_NAME}: {e}")
            raise e
