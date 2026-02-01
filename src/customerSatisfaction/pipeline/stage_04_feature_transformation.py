from customerSatisfaction.config.configuration import ConfigurationManager
from customerSatisfaction.components.feature_transformation import FeatureTransformation
from customerSatisfaction import logger

STAGE_NAME = "Feature Transformation Stage"

class FeatureTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        feature_transformation_config = config.get_feature_transformation_config()
        feature_transformation = FeatureTransformation(config=feature_transformation_config)
        feature_transformation.run_transformation()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = FeatureTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e