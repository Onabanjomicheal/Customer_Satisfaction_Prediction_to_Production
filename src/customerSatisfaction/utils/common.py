# customerSatisfaction/utils/common.py

import os
from box.exceptions import BoxValueError
import yaml
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
from customerSatisfaction import logger

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns it as a ConfigBox object.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises:
        ValueError: if the YAML file is empty.
        Exception: for any other error while reading the file.

    Returns:
        ConfigBox: YAML content as a ConfigBox object.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(paths: list, verbose=True):
    """
    Creates directories from a list of paths.

    Args:
        paths (list): List of directory paths to create.
        verbose (bool): Whether to log the creation.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """Save a dictionary as a JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Load a JSON file as a ConfigBox object."""
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """Save data as a binary file using joblib."""
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """Load data from a binary file using joblib."""
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """Return the size of a file in KB."""
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


@ensure_annotations
def decode_image(img_string: str, file_path: Path):
    """Decode a base64 image string and save it to a file."""
    img_data = base64.b64decode(img_string)
    with open(file_path, 'wb') as f:
        f.write(img_data)
    logger.info(f"Image decoded and saved at: {file_path}")


@ensure_annotations
def encode_image_to_base64(image_path: Path) -> bytes:
    """Encode an image file into a base64 string."""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read())
    return encoded
