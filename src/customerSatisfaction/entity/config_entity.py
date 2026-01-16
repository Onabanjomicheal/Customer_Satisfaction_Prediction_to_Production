from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
from box import ConfigBox

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass
class DataValidationConfig:
    root_dir: Path
    unzip_data_dir: Path
    STATUS_FILE: str
    report_file: Path
    raw_validated_dir: Path
    schema_path: Path
    full_schema: Dict[str, Any]   

