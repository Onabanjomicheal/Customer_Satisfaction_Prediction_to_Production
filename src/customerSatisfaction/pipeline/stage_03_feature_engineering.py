from pathlib import Path
from customerSatisfaction import logger
from customerSatisfaction.config.configuration import ConfigurationManager
from customerSatisfaction.components.feature_engineering import FeatureEngineering

STAGE_NAME = "Feature Engineering Stage"

class FeatureEngineeringTrainingPipeline:
    """
    Stage 03: Feature Engineering
    ----------------------------------------
    - Loads the unified Parquet file from Stage 02
    - Executes math-based transformations (deltas, bins, ratios)
    - Performs final imputation and binary labeling
    - Saves the final feature-ready dataset
    """

    def __init__(self):
        pass

    def main(self):
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        try:
            # 1️⃣ Load pipeline configuration
            config = ConfigurationManager()

            # 2️⃣ Retrieve FeatureEngineeringConfig entity
            fe_config = config.get_feature_engineering_config()

            # 3️⃣ Pre-check: Input Parquet File
            # Note: We now check for the specific file output from Stage 2
            input_file = Path(fe_config.data_path)
            output_file = Path(fe_config.engineered_data_path)

            logger.info(f"Checking for input data at: {input_file}")
            if not input_file.exists():
                raise FileNotFoundError(
                    f"Input Parquet file not found: {input_file}. "
                    "Ensure Stage 02 (Data Validation) ran successfully."
                )

            # 4️⃣ Initialize FeatureEngineering component
            fe_component = FeatureEngineering(config=fe_config)

            # 5️⃣ Run the engineering logic (Task-by-task transformation)
            # Method name updated to match our component logic
            fe_component.run_feature_engineering()

            # 6️⃣ Verify output
            if not output_file.exists():
                raise FileNotFoundError(f"Feature engineering output was not created: {output_file}")

            logger.info(
                f"Feature engineering completed successfully. "
                f"Features saved at: {output_file}"
            )
            logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        except Exception as e:
            logger.exception(f"Error in {STAGE_NAME}: {e}")
            raise e

if __name__ == "__main__":
    pipeline = FeatureEngineeringTrainingPipeline()
    pipeline.main()