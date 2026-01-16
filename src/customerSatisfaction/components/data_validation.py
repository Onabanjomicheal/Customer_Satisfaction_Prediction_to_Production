import os
import pandas as pd
import logging
from pathlib import Path
from customerSatisfaction.entity.config_entity import DataValidationConfig
from customerSatisfaction.utils.common import save_json

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.schema = config.full_schema  # schema embedded in config now
        self.report = {}  # store detailed validation results per dataset

        # Ensure validated data directory exists and assign to instance
        self.validated_data_dir = Path(self.config.raw_validated_dir)
        self.validated_data_dir.mkdir(parents=True, exist_ok=True)

    def _validate_column_types(self, data: pd.DataFrame, expected_columns: dict, dataset_name: str) -> bool:
        status = True
        for col, expected_type in expected_columns.items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                if actual_type != expected_type:
                    status = False
                    logging.error(f"Column '{col}' in {dataset_name} expected type '{expected_type}', got '{actual_type}'")
                else:
                    logging.info(f"Column '{col}' type validated in {dataset_name}")
        return status

    def _validate_constraints(self, data: pd.DataFrame, constraints: dict, dataset_name: str) -> bool:
        status = True
        if not constraints:
            return status

        # Numeric min/max checks
        for col, bound in constraints.get("min_max", {}).items():
            if col in data.columns:
                if data[col].min() < bound.get("min", float('-inf')) or data[col].max() > bound.get("max", float('inf')):
                    status = False
                    logging.error(f"Column '{col}' in {dataset_name} violates min/max constraints")

        # Allowed values checks
        for col, allowed in constraints.get("allowed_values", {}).items():
            if col in data.columns:
                invalid_values = set(data[col].unique()) - set(allowed)
                if invalid_values:
                    status = False
                    logging.error(f"Column '{col}' in {dataset_name} contains invalid values: {invalid_values}")

        # Critical columns must not be null
        for col in constraints.get("critical_columns", []):
            if col in data.columns:
                null_count = data[col].isna().sum()
                if null_count > 0:
                    status = False
                    logging.error(f"Critical column '{col}' in {dataset_name} has {null_count} nulls")

        # Max null check
        max_null = constraints.get("max_null_count")
        if max_null is not None:
            total_nulls = data.isna().sum().sum()
            if total_nulls > max_null:
                status = False
                logging.error(f"Dataset {dataset_name} exceeds max nulls: {total_nulls} > {max_null}")

        return status

    def validate_all_columns(self) -> bool:
        overall_status = True
        data_dir = Path(self.config.unzip_data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} does not exist")

        all_files = os.listdir(data_dir)
        for file_name in all_files:
            file_name_norm = file_name.strip()
            if not file_name_norm.endswith(".csv"):
                logging.info(f"Skipping non-CSV file: {file_name_norm}")
                continue

            dataset_schema = self.schema.get(file_name_norm)
            if dataset_schema is None:
                logging.info(f"Skipping {file_name_norm}: Not defined in schema.yaml")
                continue

            if "columns" not in dataset_schema:
                raise ValueError(f"'columns' key missing in schema for {file_name_norm}")

            data = pd.read_csv(data_dir / file_name_norm)
            file_status = True

            # Column existence check
            for col in dataset_schema["columns"].keys():
                if col not in data.columns:
                    file_status = False
                    logging.error(f"Column '{col}' missing in {file_name_norm}")
                else:
                    logging.info(f"Column '{col}' validated in {file_name_norm}")

            # Column type check
            file_status &= self._validate_column_types(data, dataset_schema["columns"], file_name_norm)

            # Constraints check
            file_status &= self._validate_constraints(data, dataset_schema.get("constraints", {}), file_name_norm)

            # Save validated dataset
            validated_path = self.validated_data_dir / file_name_norm
            data.to_csv(validated_path, index=False)

            # Add to report
            self.report[file_name_norm] = {
                "columns": list(data.columns),
                "status": file_status
            }

            overall_status &= file_status

        # Save JSON report
        report_path = Path(self.config.report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(report_path, self.report)

        # Write overall status
        status_file_path = Path(self.config.STATUS_FILE)
        status_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(status_file_path, 'w') as f:
            f.write(f"Validation status: {overall_status}")

        logging.info(f"Data Validation complete. Overall status: {overall_status}")
        logging.info(f"Validated datasets saved at: {self.validated_data_dir}")
        logging.info(f"Detailed report saved at: {report_path}")
        return overall_status
