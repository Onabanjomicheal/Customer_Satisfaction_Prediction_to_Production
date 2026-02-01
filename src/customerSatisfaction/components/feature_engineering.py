import pandas as pd
import numpy as np
from customerSatisfaction import logger
from customerSatisfaction.entity.config_entity import FeatureEngineeringConfig

class FeatureEngineering:
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    def run_feature_engineering(self):
        try:
            # Task 1: Load Data
            df = pd.read_parquet(self.config.data_path)
            logger.info(f"Loaded dataset for engineering. Shape: {df.shape}")

            # Task 2: Temporal Calculations (Keep - High Signal)
            df["purchase_delivery_difference"] = (
                df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
            ).dt.days
            df["estimated_actual_delivery_difference"] = (
                df["order_estimated_delivery_date"] - df["order_delivered_customer_date"]
            ).dt.days

            # Task 3: Aggressive Imputation
            cat_cols = ["product_category_name", "payment_type", "customer_state", "customer_city"]
            for col in cat_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].mode()[0])
            
            num_cols = ["product_description_lenght", "payment_value", "price", "product_weight_g"]
            for col in num_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())

            # Task 4: Binning & Target Labels
            df["labels"] = df["review_score"].apply(lambda x: 1 if x > 3 else 0)
            
            q1, q3 = df["price"].quantile([0.25, 0.75])
            df["price_category"] = df["price"].apply(
                lambda x: "expensive" if x >= q3 else ("affordable" if x >= q1 else "cheap")
            )
            
            df["purchase_delivery_diff_per_price"] = df.apply(
                lambda row: row["purchase_delivery_difference"] / row["price"] if row["price"] > 0 else 0,
                axis=1
            )

            # Task 5: DROP NON-ML COLUMNS, LEAKAGE, & HIGH-CARDINALITY NOISE
            # Senior ML Recommendation: Dropping cities, zips, and raw text 
            # to prevent overfitting and 120+ column explosion.
            drop_cols = [
                # Leaky Data & Timestamps
                "order_purchase_timestamp", "order_approved_at", 
                "order_delivered_carrier_date", "order_delivered_customer_date",
                "order_estimated_delivery_date", "shipping_limit_date",
                "review_score", "review_comment_title", "review_id",
                "review_creation_date", "review_answer_timestamp",
                "review_creation_date", "review_answer_timestamp",
                
                # Identifiers
                "customer_id", "customer_unique_id", "order_id", 
                "product_id", "seller_id", "order_item_id",

                # High-Cardinality/Low-Signal Noise
                "customer_city", "seller_city", 
                "customer_zip_code_prefix", "seller_zip_code_prefix",
                "review_comment_message", "order_status"
            ]
            
            df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
            
            # Final Row Drop for safety
            df.dropna(inplace=True)

            # Task 6: Final Feature Health Report
            logger.info("Task 6: Final Feature Health Check...")
            nan_report = (df.isna().mean() * 100).round(2)
            print("\n" + "="*50)
            print("FINAL ML-READY FEATURE HEALTH (NaN %)")
            print("="*50)
            print(nan_report[nan_report > 0].sort_values(ascending=False))
            print("Dataset Cleanliness: 100%" if df.isna().sum().sum() == 0 else "Cleanup needed")
            print("="*50 + "\n")

            # Task 7: Save
            df.to_parquet(self.config.engineered_data_path, index=False)
            logger.info(f"Stage 3 Complete. Final Shape: {df.shape}")

        except Exception as e:
            logger.exception("Feature Engineering failed")
            raise e