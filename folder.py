import os
from pathlib import Path
import logging

# Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "customerSatisfaction"

list_of_files = [

    # -------------------------
    # CI / Git
    # -------------------------
    ".github/workflows/.gitkeep",
    ".gitignore",

    # -------------------------
    # Source package
    # -------------------------
    f"src/{project_name}/__init__.py",

    # Components
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",

    # Traditional pipelines
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",

    # ZenML pipelines
    f"src/{project_name}/zenml_pipeline/__init__.py",
    f"src/{project_name}/zenml_pipeline/training_pipeline.py",
    f"src/{project_name}/zenml_pipeline/steps/__init__.py",
    f"src/{project_name}/zenml_pipeline/steps/ingest_step.py",
    f"src/{project_name}/zenml_pipeline/steps/transform_step.py",
    f"src/{project_name}/zenml_pipeline/steps/train_step.py",
    f"src/{project_name}/zenml_pipeline/steps/evaluate_step.py",

    # Config handling
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",

    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",

    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",

    # -------------------------
    # Project-level configs
    # -------------------------
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",

    # -------------------------
    # Artifacts
    # -------------------------
    "artifacts/.gitkeep",

    # -------------------------
    # Setup
    # -------------------------
    "requirements.txt",
    "setup.py",
    "README.md",

    # -------------------------
    # Research
    # -------------------------
    "research/eda.ipynb",
    "research/modeling.ipynb",

    # -------------------------
    # App
    # -------------------------
    "templates/index.html"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir:
        os.makedirs(filedir, exist_ok=True)

    if not filepath.exists() or filepath.stat().st_size == 0:
        filepath.touch()
        logging.info(f"Created: {filepath}")
    else:
        logging.info(f"Exists: {filepath}")
