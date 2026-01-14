import os
import zipfile
import gdown
import logging
from pathlib import Path
from customerSatisfaction.utils.common import get_size, create_directories
from customerSatisfaction.entity.config_entity import DataIngestionConfig

# Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        """
        Fetch data from the Google Drive URL and save as local zip file
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = Path(self.config.local_data_file)

            # Ensure the parent directory exists
            create_directories([zip_download_dir.parent])

            logging.info(f"Downloading data from {dataset_url} into {zip_download_dir}")

            # Extract file ID from Google Drive URL
            file_id = dataset_url.split("/")[-2]
            gdrive_url = f"https://drive.google.com/uc?export=download&id={file_id}"

            # Download the file
            gdown.download(url=gdrive_url, output=str(zip_download_dir), quiet=False)
            logging.info(f"Downloaded data to {zip_download_dir} (size: {get_size(zip_download_dir)})")

            return str(zip_download_dir)

        except Exception as e:
            logging.error(f"Error downloading file: {e}")
            raise e

    def extract_zip_file(self) -> str:
        """
        Extracts the zip file into the target directory
        Returns the path to the unzipped folder
        """
        try:
            unzip_path = Path(self.config.unzip_dir)
            create_directories([unzip_path])

            logging.info(f"Extracting {self.config.local_data_file} to {unzip_path}")
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)

            logging.info(f"Extraction complete. Data available at {unzip_path}")
            return str(unzip_path)

        except Exception as e:
            logging.error(f"Error extracting zip file: {e}")
            raise e

    def run_ingestion(self) -> str:
        """
        Executes the full ingestion: download + extract
        Returns the path to the raw unzipped data
        """
        self.download_file()
        return self.extract_zip_file()
