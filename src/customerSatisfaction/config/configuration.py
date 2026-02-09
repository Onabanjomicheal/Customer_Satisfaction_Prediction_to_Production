from pathlib import Path
from customerSatisfaction.utils.common import read_yaml, create_directories
from customerSatisfaction.constants import CONFIG_FILE_PATH, SCHEMA_FILE_PATH, PARAMS_FILE_PATH
from customerSatisfaction.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    FeatureEngineeringConfig,
    FeatureTransformationConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig
)

class ConfigurationManager:
    """
    Loads YAML config, params, and schema files, and provides strongly-typed
    config objects for all pipeline stages.
    """

    def __init__(
        self,
        config_filepath: str = CONFIG_FILE_PATH,
        params_filepath: str = PARAMS_FILE_PATH,
        schema_filepath: str = SCHEMA_FILE_PATH
    ):
        # Load global config, params, and schema
        self.config = read_yaml(Path(config_filepath))
        self.params = read_yaml(Path(params_filepath)) # Critical addition
        self.schema = read_yaml(Path(schema_filepath))

        # Ensure artifacts root exists
        create_directories([Path(self.config["artifacts_root"])])

    # ---------------- DATA INGESTION ---------------- #
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config["data_ingestion"]
        create_directories([Path(cfg["root_dir"])])
        return DataIngestionConfig(
            root_dir=Path(cfg["root_dir"]),
            source_URL=cfg["source_URL"],
            local_data_file=Path(cfg["local_data_file"]),
            unzip_dir=Path(cfg["unzip_dir"])
        )

    # ---------------- DATA VALIDATION ---------------- #
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema 

        create_directories([config.root_dir, config.raw_validated_dir])

        return DataValidationConfig(
            root_dir=Path(config.root_dir),
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir=Path(config.unzip_data_dir),
            raw_validated_dir=Path(config.raw_validated_dir),
            report_file=Path(config.report_file),
            datasets=config.datasets,
            all_schema=schema
        )

    # ---------------- FEATURE ENGINEERING ---------------- #
    def get_feature_engineering_config(self) -> FeatureEngineeringConfig:
        config = self.config.feature_engineering
        target_name = self.schema.target_configuration.prediction_target

        create_directories([config.root_dir])

        return FeatureEngineeringConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            engineered_data_path=Path(config.engineered_data_path),
            target_column=str(target_name)
        )

    # ---------------- FEATURE TRANSFORMATION ---------------- #
    def get_feature_transformation_config(self) -> FeatureTransformationConfig:
        config = self.config.feature_transformation

        create_directories([config.root_dir])

        return FeatureTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            transformer_path=Path(config.transformer_path),
            transformed_train_path=Path(config.transformed_train_path),
            transformed_test_path=Path(config.transformed_test_path),
            test_size=config.test_size,
            random_state=config.random_state
        )

    # ---------------- MODEL TRAINING ---------------- #
    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        model_params = self.params.models  


        target_col = self.schema.target_configuration.prediction_target
        create_directories([config.root_dir])

        return ModelTrainingConfig(
            root_dir=Path(config.root_dir),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            model_name=config.model_name,
            model_path=Path(config.model_path),
            all_params=model_params,
            target_column=target_col
        )
    
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        
        # FIX: Just take the whole params block. 
        # Don't ask for .RandomForest specifically.
        all_params = self.params 

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data_path=Path(config.test_data_path),
            model_path=Path(config.model_path),
            all_params=all_params,
            metric_file_name=Path(config.metric_file_name),
            target_column=config.target_column,
            mlflow_uri=config.mlflow_uri
        )

        return model_evaluation_config