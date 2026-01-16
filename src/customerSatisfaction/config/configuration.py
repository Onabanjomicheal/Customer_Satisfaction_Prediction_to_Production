from pathlib import Path
from customerSatisfaction.utils.common import read_yaml, create_directories
from customerSatisfaction.entity.config_entity import DataIngestionConfig, DataValidationConfig
from customerSatisfaction.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH

class ConfigurationManager:
    """
    Loads YAML files and provides DataIngestionConfig and DataValidationConfig.
    Schema is now included inside DataValidationConfig for uniform access.
    """

    def __init__(self,
                 config_filepath: str = CONFIG_FILE_PATH,
                 params_filepath: str = PARAMS_FILE_PATH,
                 schema_filepath: str = SCHEMA_FILE_PATH):
        self.config = dict(read_yaml(config_filepath))   # plain dict
        self.params = dict(read_yaml(params_filepath))
        self.schema = dict(read_yaml(schema_filepath))

        # Ensure artifacts root exists
        create_directories([self.config.get('artifacts_root', 'artifacts')])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config['data_ingestion']
        create_directories([config['root_dir']])
        return DataIngestionConfig(
            root_dir=Path(config['root_dir']),
            source_URL=config['source_URL'],
            local_data_file=Path(config['local_data_file']),
            unzip_dir=Path(config['unzip_dir'])
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Returns DataValidationConfig with schema embedded.
        """
        config = self.config['data_validation']
        create_directories([config['root_dir'], config['raw_validated_dir']])
        return DataValidationConfig(
            root_dir=Path(config['root_dir']),
            unzip_data_dir=Path(config['unzip_data_dir']),
            STATUS_FILE=str(config['STATUS_FILE']),
            report_file=Path(config['report_file']),
            raw_validated_dir=Path(config['raw_validated_dir']),
            schema_path=Path(config['schema_path']),
            full_schema=self.schema  # embed the schema directly
        )
