import argparse
from customerSatisfaction import logger 
from customerSatisfaction.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from customerSatisfaction.pipeline.stage_02_data_validation import DataValidationPipeline
from customerSatisfaction.pipeline.stage_03_feature_engineering import FeatureEngineeringTrainingPipeline
from customerSatisfaction.pipeline.stage_04_feature_transformation import FeatureTransformationTrainingPipeline
from customerSatisfaction.pipeline.stage_05_model_training import ModelTrainingPipeline
from customerSatisfaction.pipeline.stage_06_model_evaluation import ModelEvaluationPipeline

# Stage execution map
STAGE_MAP = {
    "data_ingestion": DataIngestionTrainingPipeline,
    "data_validation": DataValidationPipeline,
    "feature_engineering": FeatureEngineeringTrainingPipeline,
    "feature_transformation": FeatureTransformationTrainingPipeline,
    "model_training": ModelTrainingPipeline,
    "model_evaluation": ModelEvaluationPipeline,
}

def run_stage(stage_name: str):
    try:
        logger.info(f"\n\n{'='*30}\nSTAGE: {stage_name.upper()}\n{'='*30}")
        pipeline = STAGE_MAP[stage_name]()
        pipeline.main()
        logger.info(f"\n\n{'='*30}\n{stage_name.upper()} COMPLETED\n{'='*30}\n\nx==========x")
    except Exception as e:
        logger.exception(f"Exception occurred in {stage_name}: {e}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Satisfaction Prediction Pipeline")
    choices = list(STAGE_MAP.keys()) + ["all"]
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=choices,
        help="The pipeline stage to execute, or 'all' to run the full pipeline"
    )
    args = parser.parse_args()

    if args.stage == "all":
        logger.info("Starting Full Pipeline Execution...")
        for stage in STAGE_MAP.keys():
            run_stage(stage)
    else:
        run_stage(args.stage)
