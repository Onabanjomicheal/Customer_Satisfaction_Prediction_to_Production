from pathlib import Path
import os
from customerSatisfaction.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from customerSatisfaction.utils.common import read_yaml, create_directories, save_json
from customerSatisfaction.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    TrainingConfig,
    EvaluationConfig
)


class ConfigurationManager:
    def __init__(self, config_filepath="config/config.yaml", params_filepath="params.yaml"):
        self.config = read_yaml(Path(config_filepath))
        self.params = read_yaml(Path(params_filepath))

        # Ensure artifacts root exists
        create_directories([self.config['artifacts_root']])

    # -------------------------
    # Stage 01: Data Ingestion
    # -------------------------
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config['data_ingestion']
        create_directories([config['root_dir']])

        return DataIngestionConfig(
            root_dir=Path(config['root_dir']),
            source_URL=config['source_URL'],
            local_data_file=Path(config['local_data_file']),
            unzip_dir=Path(config['unzip_dir'])
        )

    # -------------------------
    # Stage 02: Data Validation
    # -------------------------
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config['data_validation']
        create_directories([config['root_dir']])

        return DataValidationConfig(
            root_dir=Path(config['root_dir']),
            raw_data_dir=Path(self.config['data_ingestion']['unzip_dir']),
            validated_data_file=Path(config['validated_data_file'])
        )


    # -------------------------
    # Stage 03: Data Transformation
    # -------------------------
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config['data_transformation']
        create_directories([config['root_dir']])

        return DataTransformationConfig(
            root_dir=Path(config['root_dir']),
            transformed_train_file=Path(config['transformed_train_file']),
            transformed_test_file=Path(config['transformed_test_file']),
            preprocessor_object_file=Path(config['preprocessor_object_file'])
        )

    # -------------------------
    # Stage 04: Training
    # -------------------------
    def get_training_config(self) -> TrainingConfig:
        config = self.config['training']
        create_directories([config['root_dir']])

        return TrainingConfig(
            root_dir=Path(config['root_dir']),
            trained_model_file=Path(config['trained_model_file']),
            model_config=self.params['MODEL_CONFIG'],
            training_data=Path(config['training_data'])
        )

    # -------------------------
    # Stage 05: Evaluation
    # -------------------------
    def get_evaluation_config(self) -> EvaluationConfig:
        config = self.config['evaluation']
        create_directories([config['root_dir']])

        return EvaluationConfig(
            root_dir=Path(config['root_dir']),
            trained_model_file=Path(config['trained_model_file']),
            test_data=Path(config['test_data']),
            evaluation_report_file=Path(config['evaluation_report_file']),
            metrics=None
        )
