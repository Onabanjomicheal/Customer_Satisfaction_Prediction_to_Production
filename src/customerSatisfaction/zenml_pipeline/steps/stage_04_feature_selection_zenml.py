from zenml import step
from pathlib import Path
from customerSatisfaction.config.configuration import ConfigurationManager
from customerSatisfaction.components.feature_selection import FeatureSelection
from customerSatisfaction.utils.common import logger


@step(enable_cache=True)
def feature_selection_step() -> str:
    """
    ZenML Step: Feature Selection
    -----------------------------
    - Loads feature-engineered dataset
    - Runs feature selection using existing component
    - Returns path to selected features file
    """

    logger.info("ZenML Step: Feature Selection started")

    # 1️⃣ Load configuration
    config_manager = ConfigurationManager()
    feature_selection_config = config_manager.get_feature_selection_config()

    # 2️⃣ Ensure artifact directory exists
    Path(feature_selection_config.root_dir).mkdir(parents=True, exist_ok=True)

    # 3️⃣ Run Feature Selection component (business logic)
    selector = FeatureSelection(config=feature_selection_config)
    selector.run()

    logger.info(
        f"Feature Selection completed. "
        f"Selected features saved at: {feature_selection_config.selected_features_file}"
    )

    # 4️⃣ Return artifact path (ZenML can track this)
    return feature_selection_config.selected_features_file
