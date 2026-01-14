from dataclasses import dataclass
from pathlib import Path

# -------------------------------
# Data Ingestion Config
# -------------------------------
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path          # artifacts/data_ingestion
    source_URL: str         # Google Drive link
    local_data_file: Path   # artifacts/data_ingestion/data.zip
    unzip_dir: Path         # artifacts/data_ingestion/customer_data

# -------------------------------
# Data Validation Config
# -------------------------------
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    raw_data_dir: Path
    validated_data_file: Path       # artifacts/data_validation/validated_data.csv

# -------------------------------
# Data Transformation Config
# -------------------------------
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    transformed_train_file: Path
    transformed_test_file: Path
    preprocessor_object_file: Path  # e.g., preprocessing pipeline pickle

# -------------------------------
# Model Training Config
# -------------------------------
@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_file: Path
    model_config: dict              # e.g., {"n_estimators":100, "max_depth":7, ...}
    training_data: Path             # path to transformed training data

# -------------------------------
# Model Evaluation Config
# -------------------------------
@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    trained_model_file: Path
    test_data: Path                 # path to transformed test data
    evaluation_report_file: Path
    metrics: dict = None            # optional default metrics container
