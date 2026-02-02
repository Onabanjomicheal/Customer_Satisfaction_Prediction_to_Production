from dataclasses import dataclass
from pathlib import Path


# ---------------- DATA INGESTION ---------------- #
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


# ---------------- DATA VALIDATION ---------------- #
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    raw_validated_dir: Path
    report_file: Path
    datasets: list
    all_schema: dict


# ---------------- FEATURE ENGINEERING ---------------- #
@dataclass(frozen=True)
class FeatureEngineeringConfig:
    root_dir: Path
    data_path: Path
    engineered_data_path: Path
    target_column: str 


# ---------------- FEATURE TRANSFORMATION ---------------- #
@dataclass(frozen=True)
class FeatureTransformationConfig:
    root_dir: Path
    data_path: Path
    transformer_path: Path
    transformed_train_path: Path
    transformed_test_path: Path
    test_size: float
    random_state: int



# ---------------- MODEL TRAINING ---------------- #
@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str       
    model_path: Path
    all_params: dict
    mlflow_uri: str = None

# ---------------- MODEL EVALUATION ---------------- #
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str  # 
