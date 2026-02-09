import pandas as pd
import numpy as np
from customerSatisfaction import logger
from customerSatisfaction.entity.config_entity import FeatureEngineeringConfig

class FeatureEngineering:
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    def run_feature_engineering(self):
        try:
            # Load validated data
            df = pd.read_parquet(self.config.data_path)
            logger.info(f"Dataset loaded. Shape: {df.shape}")
            
            # ========================================
            # 0. SCHEMA ALIGNMENT & CLEANING
            # ========================================
            # Mapping common Olist names to your expected feature names
            rename_map = {
                'freight_value': 'total_freight',
                'price': 'total_price',
                'order_item_id': 'order_items_count'
            }
            
            for old_col, new_col in rename_map.items():
                if old_col in df.columns and new_col not in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)
                    logger.info(f"Renamed column '{old_col}' to '{new_col}'")

            # Final check for critical columns
            required_cols = ['total_freight', 'total_price', 'order_purchase_timestamp']
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                raise KeyError(f"Critical columns missing: {missing_cols}")

            logger.info("="*80)
            logger.info("STAGE 3: PRODUCTION FEATURE ENGINEERING (NG-STYLE)")
            logger.info("="*80)

            # Convert to datetime
            datetime_cols = [
                'order_purchase_timestamp', 'order_approved_at', 
                'order_delivered_carrier_date', 'order_delivered_customer_date',
                'order_estimated_delivery_date'
            ]
            for col in datetime_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')

            # ========================================
            # 1. PSYCHOLOGICAL DELIVERY FEATURES
            # ========================================
            logger.info("1. Creating Psychological Delivery Features...")
            
            # Actual total delivery time
            df['delivery_time_days'] = (
                (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.total_seconds() / (24 * 3600)
            ).fillna(0)
            
            # Promised delivery time
            df['estimated_delivery_days'] = (
                (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.total_seconds() / (24 * 3600)
            ).fillna(0)
            
            # Expectation Gap (Positive = Late, Negative = Early)
            df['delivery_expectation_gap'] = df['delivery_time_days'] - df['estimated_delivery_days']
            
            # Delay Severity (Non-linear penalty for extreme lateness)
            df['delay_severity'] = np.where(df['delivery_expectation_gap'] > 0, df['delivery_expectation_gap']**2, 0)
            
            if 'order_delivered_carrier_date' in df.columns and 'order_approved_at' in df.columns:
                df['carrier_handling_time'] = (
                    (df['order_delivered_carrier_date'] - df['order_approved_at']).dt.total_seconds() / (24 * 3600)
                ).fillna(0)
            
            df['is_late_delivery'] = (df['delivery_expectation_gap'] > 0).astype(int)

            # ========================================
            # 2. PRICING & VALUE SENSITIVITY
            # ========================================
            logger.info("2. Creating Value Sensitivity Features...")
            
            # Freight Ratio: High shipping costs relative to price trigger higher expectations
            df['freight_ratio'] = (df['total_freight'] / (df['total_price'] + df['total_freight'])).fillna(0)
            
            # High Value Flag: Customers spending more than the median have lower tolerance for errors
            df['is_high_value'] = (df['total_price'] > df['total_price'].median()).astype(int)
            
            if 'order_items_count' in df.columns:
                df['avg_item_price'] = df['total_price'] / df['order_items_count']
            
            df['used_installments'] = (df['payment_installments'] > 1).astype(int) if 'payment_installments' in df.columns else 0

            # ========================================
            # 3. TEMPORAL (SEASONALITY) FEATURES
            # ========================================
            logger.info("3. Creating Temporal Features...")
            
            df['order_day_of_week'] = df['order_purchase_timestamp'].dt.dayofweek
            df['order_hour'] = df['order_purchase_timestamp'].dt.hour
            df['order_month'] = df['order_purchase_timestamp'].dt.month
            df['is_weekend_order'] = (df['order_day_of_week'] >= 5).astype(int)

            # ========================================
            # 4. TARGET VARIABLE & CLEANUP
            # ========================================
            logger.info("4. Finalizing Target and Cleanup...")
            
            if 'review_score' in df.columns:
                # Binary classification: 1 = Happy (4,5), 0 = Not Happy (1,2,3)
                df['is_satisfied'] = (df['review_score'] >= 4).astype(int)
            
            # Drop columns that cannot be used in a prediction service (like raw IDs/Dates)
            drop_cols = datetime_cols + ['customer_id', 'order_id', 'review_score', 'order_status']
            existing_drop_cols = [c for c in drop_cols if c in df.columns]
            df.drop(columns=existing_drop_cols, inplace=True)

            # Standard Data-Centric Cleanup: Handle missing values with medians
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

            # Save engineered data
            df.to_parquet(self.config.engineered_data_path, index=False)
            
            # Final output log and print of existing features
            logger.info(f"STAGE 3 COMPLETE. Final Shape: {df.shape}")
            print("-" * 30)
            print("EXISTING FEATURES AFTER CLEANUP:")
            print(df.columns.tolist())
            print("-" * 30)

        except Exception as e:
            logger.exception("Feature Engineering failed")
            raise e