from customerSatisfaction import logger
from customerSatisfaction.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from customerSatisfaction.pipeline.stage_02_data_validation import DataValidationPipeline

def run_stage(stage_name: str, pipeline):
    """Helper to run any stage with logging and exception handling"""
    try:
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")
        pipeline.main()
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(f"Exception occurred in {stage_name}: {e}")
        raise e

def run_pipeline():
    run_stage("Data Ingestion stage", DataIngestionTrainingPipeline())
    run_stage("Data Validation stage", DataValidationPipeline())

if __name__ == "__main__":
    run_pipeline()
