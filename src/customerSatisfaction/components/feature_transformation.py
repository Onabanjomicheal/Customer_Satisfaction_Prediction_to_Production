import pandas as pd
import numpy as np
import joblib
import os
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
            logger.info("="*80)
            logger.info("STAGE 4: FEATURE TRANSFORMATION")
            logger.info("="*80)
            
            # 1. LOAD ENGINEERED DATA
            logger.info("1. Loading Engineered Features...")
            df = pd.read_parquet(self.config.data_path)
            logger.info(f"    [OK] Loaded data: {df.shape}")

            # 2. SEPARATE FEATURES AND TARGET
            logger.info("2. Separating Features and Target...")
            target_col = 'is_satisfied'
            
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in dataset!")
            
            X = df.drop(columns=[target_col])
            y = df[target_col].astype(int)
            
            logger.info(f"    [OK] Features (X): {X.shape}")
            logger.info(f"    [OK] Target (y): {y.shape}")
            logger.info(f"    [OK] Target distribution: {y.value_counts().to_dict()}")

            # 3. STRATIFIED TRAIN-TEST SPLIT
            logger.info("3. Creating Stratified Train-Test Split...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config.test_size, 
                stratify=y, 
                random_state=self.config.random_state
            )
            
            logger.info(f"    [OK] Train set: {X_train.shape[0]:,} samples")
            logger.info(f"    [OK] Test set: {X_test.shape[0]:,} samples")

            # 4. DEFINE FEATURE GROUPS
            logger.info("4. Defining Feature Groups...")
            
            # Numeric Feature logic (Standardizing Olist features)
            delivery_features = ['delivery_time_days', 'estimated_delivery_days', 'delivery_delay_days', 'is_late_delivery', 'is_early_delivery']
            if 'carrier_handling_time' in X.columns:
                delivery_features.extend(['carrier_handling_time', 'carrier_to_customer_time'])
            
            pricing_features = ['total_price', 'avg_price', 'max_price', 'min_price', 'total_freight', 'avg_freight', 'max_freight', 'freight_to_price_ratio', 'avg_item_price', 'price_range', 'payment_value', 'payment_price_diff', 'payment_installments', 'used_installments', 'high_installments']
            
            product_features = ['order_items_count', 'total_weight_g', 'avg_weight_g', 'max_weight_g', 'avg_length_cm', 'max_length_cm', 'avg_height_cm', 'max_height_cm', 'avg_width_cm', 'max_width_cm', 'product_volume_cm3', 'weight_per_item', 'product_density', 'is_heavy_item', 'is_bulky_item', 'is_multi_item']
            
            if 'product_photos_qty' in X.columns: product_features.append('product_photos_qty')
            if 'product_description_lenght' in X.columns: product_features.append('product_description_lenght')
            
            temporal_features = ['order_day_of_week', 'order_hour', 'order_month', 'is_weekend_order', 'is_business_hours', 'is_holiday_season']
            
            payment_behavior_features = ['is_credit_card', 'is_boleto', 'is_voucher', 'is_debit_card', 'multiple_payments']
            if 'num_payments' in X.columns: payment_behavior_features.append('num_payments')
            
            # Categorical Mapping
            categorical_features = [f for f in ['product_category_name', 'payment_type', 'customer_state'] if f in X.columns]
            
            # Compile all numeric
            numeric_features = []
            for f_list in [delivery_features, pricing_features, product_features, temporal_features, payment_behavior_features]:
                numeric_features.extend([f for f in f_list if f in X.columns])
            numeric_features = list(set(numeric_features))
            
            logger.info(f"    [OK] Numeric features: {len(numeric_features)}")
            logger.info(f"    [OK] Categorical features: {len(categorical_features)}")

            # 5. BUILD PREPROCESSING PIPELINE
            logger.info("5. Building Preprocessing Pipeline...")
            transformers = []
            if numeric_features:
                transformers.append(("num", StandardScaler(), numeric_features))
                logger.info(f"    [OK] StandardScaler active for numeric features")
            
            if categorical_features:
                transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features))
                logger.info(f"    [OK] OneHotEncoder active for categorical features")
            
            preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
            
            # 6. FIT PREPROCESSOR
            logger.info("6. Fitting Preprocessor...")
            preprocessor.fit(X_train)
            logger.info("    [OK] Preprocessor fitted on training data")
            
            try:
                feature_names_out = preprocessor.get_feature_names_out()
                logger.info(f"    [OK] Transformation output features: {len(feature_names_out)}")
            except:
                logger.info("    [INFO] Output feature names not retrievable")

            # 7. PREPARE DATA FOR STORAGE
            logger.info("7. Preparing Data for Storage...")
            train_data = X_train.copy()
            train_data['target'] = y_train.values
            test_data = X_test.copy()
            test_data['target'] = y_test.values
            
            # 8. SAVE ARTIFACTS
            logger.info("8. Saving Transformation Artifacts...")
            joblib.dump(preprocessor, self.config.transformer_path)
            train_data.to_csv(self.config.transformed_train_path, index=False)
            test_data.to_csv(self.config.transformed_test_path, index=False)
            
            logger.info(f"    [OK] Preprocessor and CSVs saved to artifacts.")

            # 9. VALIDATION CHECK
            logger.info("9. Validation Check...")
            try:
                X_train_transformed = preprocessor.transform(X_train.head(5))
                logger.info(f"    [OK] Preprocessor test successful. Output shape: {X_train_transformed.shape}")
            except Exception as e:
                logger.warning(f"    [WARN] Preprocessor test failed: {str(e)}")

            # 10. SUMMARY
            print("\n" + "="*80)
            print("STAGE 4 TRANSFORMATION SUMMARY")
            print("="*80)
            print(f"Original Features: {X.shape[1]} -> Transformed: {len(feature_names_out) if 'feature_names_out' in locals() else 'N/A'}")
            print(f"Data Split: Training ({len(train_data):,}) | Testing ({len(test_data):,})")
            print("="*80 + "\n")

            logger.info("[OK] STAGE 4 COMPLETE - Feature Transformation Successful")
            logger.info("="*80)

        except Exception as e:
            logger.exception("Feature Transformation failed")
            raise e