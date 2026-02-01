import os
import json
import joblib
import yaml
import logging
from pathlib import Path
from typing import Any
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
import base64
import pandas as pd

# ---------------- LOGGING ---------------- #
from customerSatisfaction import logger


# ---------------- YAML ---------------- #
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML loaded: {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e


# ---------------- DIRECTORIES ---------------- #
@ensure_annotations
def create_directories(paths: list, verbose=True):
    for path in paths:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created: {path}")


# ---------------- JSON ---------------- #
@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON saved: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON loaded: {path}")
    return ConfigBox(content)


# ---------------- BINARY ---------------- #
@ensure_annotations
def save_bin(data, path: Path):
    """
    Saves a pandas DataFrame or any object to disk as a binary file using joblib.
    
    Args:
        data: pandas DataFrame or object to save.
        path: Path to save the file.
    """
    try:
        # Ensure parent directories exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Optional type check
        if isinstance(data, pd.DataFrame):
            joblib.dump(data, path)
        else:
            joblib.dump(data, path)

        logger.info(f"Binary file saved: {path}")

    except Exception as e:
        logger.exception(f"Failed to save binary file at {path}: {e}")
        raise e


@ensure_annotations
def load_bin(path: Path):
    """
    Loads a binary file (pandas DataFrame or object) from disk using joblib.
    
    Args:
        path: Path to load the file from.
    Returns:
        Loaded object (usually pandas DataFrame)
    """
    try:
        data = joblib.load(path)
        logger.info(f"Binary file loaded: {path}")
        return data
    except Exception as e:
        logger.exception(f"Failed to load binary file at {path}: {e}")
        raise e


# ---------------- FILE INFO ---------------- #
@ensure_annotations
def get_size(path: Path) -> str:
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


# ---------------- IMAGE ---------------- #
@ensure_annotations
def decode_image(img_string: str, file_path: Path):
    img_data = base64.b64decode(img_string)
    with open(file_path, "wb") as f:
        f.write(img_data)
    logger.info(f"Image decoded and saved: {file_path}")


@ensure_annotations
def encode_image_to_base64(image_path: Path) -> bytes:
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read())
    return encoded
