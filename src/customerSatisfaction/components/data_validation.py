import logging
import pandas as pd
from pathlib import Path
from typing import List
from customerSatisfaction.entity.config_entity import DataValidationConfig
from customerSatisfaction.utils.common import create_directories, get_size, save_json
from customerSatisfaction.config.configuration import ConfigurationManager

# Logging setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


class DataValidation:
    """
    Stage 02: Data Validation
    """

    def __init__(self, config: DataValidationConfig):
        self.config = config
        create_directories([self.config.root_dir])

    def _log_columns(self, df: pd.DataFrame, file_name: str) -> None:
        """Log the columns of a single CSV file"""
        logging.info(f"Columns in {file_name}: {list(df.columns)}")

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicates and fill missing numeric values"""
        before_rows = len(df)
        df = df.drop_duplicates()
        logging.info(f"Dropped {before_rows - len(df)} duplicate rows")

        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                logging.info(f"Filled missing values in '{col}' with median={median_value}")

        return df

    def run_validation(self) -> List[Path]:
        """Perform full validation and return validated file paths"""
        raw_data_dir = Path(self.config.raw_data_dir)
        csv_files = list(raw_data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {raw_data_dir}")

        logging.info(f"Found {len(csv_files)} CSV files for validation")

        validated_files = []
        cleaned_dfs = []

        for csv_file in csv_files:
            logging.info(f"Validating file: {csv_file.name}")
            df = pd.read_csv(csv_file)

            if df.empty:
                logging.warning(f"{csv_file.name} is empty")
                continue

            # Log columns
            self._log_columns(df, csv_file.name)

            # Clean data
            cleaned_df = self._clean_dataframe(df)

            output_file = Path(self.config.root_dir) / f"{csv_file.stem}_cleaned.csv"
            cleaned_df.to_csv(output_file, index=False)
            validated_files.append(output_file)
            cleaned_dfs.append(cleaned_df)

            logging.info(f"Saved cleaned file: {output_file} (size: {get_size(output_file)})")

        # Merge all CSVs if needed (optional)
        if cleaned_dfs:
            merged_df = pd.concat(cleaned_dfs, ignore_index=True)
            merged_df.to_csv(self.config.validated_data_file, index=False)
            logging.info(f"Merged validated dataset saved to {self.config.validated_data_file} "
                         f"(size: {get_size(self.config.validated_data_file)})")

        # Save validation status
        status = {
            "status": "success",
            "num_files_processed": len(validated_files),
            "validated_files": [str(f) for f in validated_files],
            "merged_file": str(self.config.validated_data_file) if cleaned_dfs else None
        }
        save_json(path=Path(self.config.root_dir) / "validation_status.json", data=status)

        logging.info("Data Validation stage completed successfully")
        return validated_files


class DataValidationPipeline:
    """Wrapper pipeline for stage 2"""
    def __init__(self):
        self.config = ConfigurationManager().get_data_validation_config()
        self.validator = DataValidation(self.config)

    def main(self) -> List[Path]:
        return self.validator.run_validation()
