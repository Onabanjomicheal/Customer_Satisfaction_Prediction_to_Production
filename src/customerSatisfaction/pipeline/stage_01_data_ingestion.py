from customerSatisfaction.config.configuration import ConfigurationManager
from customerSatisfaction.components.data_ingestion import DataIngestion
import logging

STAGE_NAME = "Data Ingestion Stage"

# Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        # 1️⃣ Load configuration
        config = ConfigurationManager()

        # 2️⃣ Get DataIngestionConfig entity
        data_ingestion_config = config.get_data_ingestion_config()

        # 3️⃣ Run the ingestion component
        data_ingestion = DataIngestion(config=data_ingestion_config)
        raw_data_path = data_ingestion.run_ingestion()

        logging.info(f"Data ingestion completed. Raw data available at: {raw_data_path}")


if __name__ == "__main__":
    try:
        logging.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        data_ingestion = DataIngestionTrainingPipeline()
        data_ingestion.main()
        logging.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(f"Error in {STAGE_NAME}: {e}")
        raise e