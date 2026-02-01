# stage_01_data_ingestion_zenml.py

from zenml.steps import step
from customerSatisfaction.config.configuration import ConfigurationManager
from customerSatisfaction.components.data_ingestion import DataIngestion
from customerSatisfaction import logger

@step
def data_ingestion_step() -> str:
    """
    ZenML step for data ingestion.
    Returns the path to the raw data folder.
    """
    logger.info("Running ZenML Step: Data Ingestion")

    # Load configuration
    config_manager = ConfigurationManager()
    data_ingestion_config = config_manager.get_data_ingestion_config()

    # Run ingestion component
    ingestion_component = DataIngestion(config=data_ingestion_config)
    raw_data_path = ingestion_component.run_ingestion()

    logger.info(f"Data ingestion completed. Raw data at: {raw_data_path}")
    return str(raw_data_path)
