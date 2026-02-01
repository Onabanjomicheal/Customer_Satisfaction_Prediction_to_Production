import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from customerSatisfaction import logger
from customerSatisfaction.entity.config_entity import FeatureTransformationConfig

class FeatureTransformation:
    def __init__(self, config: FeatureTransformationConfig):
        self.config = config

    def run_transformation(self):
        try:
            # Task 1: Load Data from Stage 3
            df = pd.read_parquet(self.config.data_path)
            logger.info(f"Task 1: Loaded features. Shape: {df.shape}")

            # Task 2: Separate Features and Target
            X = df.drop(columns=["labels"])
            y = df["labels"].astype(int)

            # Task 3: Stratified Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config.test_size, 
                stratify=y, 
                random_state=self.config.random_state
            )

            # Task 4: Define Feature Groups
            numeric_features = [
                "price", "freight_value", "product_photos_qty", "product_weight_g",
                "product_length_cm", "product_height_cm", "product_width_cm",
                "payment_value", "payment_installments", "purchase_delivery_difference",
                "estimated_actual_delivery_difference", "purchase_delivery_diff_per_price"
            ]
            categorical_features = [
                "product_category_name", "price_category", "payment_type", "customer_state"
            ]
            binary_features = ["review_availability"]

            # Task 5: Build Preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numeric_features),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
                    ("bin", "passthrough", binary_features)
                ],
                remainder="drop"
            )

            # Task 6: FIT ONLY (The Fix)
            # We fit on X_train to learn scales/categories, but do NOT transform yet.
            # This keeps the data in DataFrame format for saving.
            preprocessor.fit(X_train)
            logger.info("Task 6: Preprocessor fitted. Logic saved to object.")

            # Task 7: Prepare Raw DataFrames
            # We use the original split data (X_train/X_test) which still has column names.
            train_data = X_train.copy()
            train_data['target'] = y_train.values
            
            test_data = X_test.copy()
            test_data['target'] = y_test.values
            logger.info("Task 7: Raw features preserved with column headers.")

            # Task 8: Save Artifacts
            # 1. Save the fitted transformer object (The "Translator")
            joblib.dump(preprocessor, self.config.transformer_path)
            
            # 2. Save the RAW CSVs (Will look like the table you sent, not numpy arrays)
            train_data.to_csv(self.config.transformed_train_path, index=False)
            test_data.to_csv(self.config.transformed_test_path, index=False)
            
            logger.info(f"Task 8: Saved transformer to {self.config.transformer_path}")
            logger.info(">>>>>> Stage 4 Transformation Finalized Successfully <<<<<<")

        except Exception as e:
            logger.exception("Feature Transformation failed")
            raise e