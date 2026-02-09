import pandas as pd
from pathlib import Path
from customerSatisfaction.entity.config_entity import DataValidationConfig
from customerSatisfaction.utils.common import save_json
from customerSatisfaction import logger

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.schema = config.all_schema
        self.report = {}
        self.raw_data_dir = Path(config.unzip_data_dir)
        self.validated_data_dir = Path(config.raw_validated_dir)
        self.validated_data_dir.mkdir(parents=True, exist_ok=True)

    def initiate_data_validation(self) -> bool:
        try:
            logger.info("Stage 2: Loading datasets...")
            datasets = {}
            for file_name in self.config.datasets:
                path = self.raw_data_dir / file_name
                df = pd.read_csv(path)
                datasets[file_name.replace(".csv", "")] = df
                logger.info(f"Loaded {file_name}: {df.shape}")

            # Extract datasets
            customers = datasets["olist_customers_dataset"]
            orders = datasets["olist_orders_dataset"]
            order_items = datasets["olist_order_items_dataset"]
            payments = datasets["olist_order_payments_dataset"]
            products = datasets["olist_products_dataset"]
            reviews = datasets["olist_order_reviews_dataset"]
            sellers = datasets["olist_sellers_dataset"]

            logger.info("Stage 2: Data Cleaning and Validation...")
            
            # Drop duplicates from key tables
            orders.drop_duplicates(subset=["order_id"], inplace=True)
            customers.drop_duplicates(subset=["customer_id"], inplace=True)
            products.drop_duplicates(subset=["product_id"], inplace=True)
            sellers.drop_duplicates(subset=["seller_id"], inplace=True)
            
            # Handle payment duplicates - keep highest payment value
            payments = payments.sort_values("payment_value", ascending=False).drop_duplicates(subset=["order_id"])

            # Convert datetime columns
            logger.info("Converting datetime columns...")
            datetime_cols = [
                'order_purchase_timestamp',
                'order_approved_at',
                'order_delivered_carrier_date',
                'order_delivered_customer_date',
                'order_estimated_delivery_date'
            ]
            
            for col in datetime_cols:
                if col in orders.columns:
                    orders[col] = pd.to_datetime(orders[col], errors='coerce')
            
            # Convert review datetime
            if 'review_creation_date' in reviews.columns:
                reviews['review_creation_date'] = pd.to_datetime(reviews['review_creation_date'], errors='coerce')

            # CRITICAL: Drop orders with missing critical dates BEFORE merge
            orders.dropna(subset=['order_purchase_timestamp', 'order_delivered_customer_date'], inplace=True)
            logger.info(f"Orders after date validation: {len(orders)}")

            logger.info("Stage 2: Strategic Dataset Merging...")
            
            # Step 1: Aggregate order items per order
            order_items_agg = order_items.groupby('order_id').agg({
                'order_item_id': 'count',  # Number of items
                'price': ['sum', 'mean', 'max', 'min'],
                'freight_value': ['sum', 'mean', 'max']
            }).reset_index()
            
            order_items_agg.columns = ['order_id', 'order_items_count', 
                                        'total_price', 'avg_price', 'max_price', 'min_price',
                                        'total_freight', 'avg_freight', 'max_freight']
            
            # Step 2: Aggregate payments per order
            payments_agg = payments.groupby('order_id').agg({
                'payment_sequential': 'max',
                'payment_type': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
                'payment_installments': 'max',
                'payment_value': 'sum'
            }).reset_index()
            
            payments_agg.columns = ['order_id', 'num_payments', 'payment_type', 
                                   'payment_installments', 'payment_value']
            
            # Step 3: Get product information (from first item in order for simplicity)
            order_items_products = order_items.merge(
                products[['product_id', 'product_category_name', 'product_weight_g', 
                         'product_length_cm', 'product_height_cm', 'product_width_cm',
                         'product_photos_qty', 'product_description_lenght']],
                on='product_id',
                how='left'
            )
            
            # Aggregate product features per order
            product_agg = order_items_products.groupby('order_id').agg({
                'product_category_name': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown',
                'product_weight_g': ['sum', 'mean', 'max'],
                'product_length_cm': ['mean', 'max'],
                'product_height_cm': ['mean', 'max'],
                'product_width_cm': ['mean', 'max'],
                'product_photos_qty': 'mean',
                'product_description_lenght': 'mean'
            }).reset_index()
            
            product_agg.columns = ['order_id', 'product_category_name', 
                                  'total_weight_g', 'avg_weight_g', 'max_weight_g',
                                  'avg_length_cm', 'max_length_cm',
                                  'avg_height_cm', 'max_height_cm',
                                  'avg_width_cm', 'max_width_cm',
                                  'product_photos_qty', 'product_description_lenght']
            
            # Step 4: Start merging with orders as base (INNER JOIN)
            merged_df = orders.copy()
            logger.info(f"Base orders: {len(merged_df)}")
            
            # Merge reviews (INNER - we only want orders with reviews for supervised learning)
            merged_df = merged_df.merge(
                reviews[['order_id', 'review_score', 'review_creation_date']], 
                on='order_id', 
                how='inner'
            )
            logger.info(f"After reviews merge: {len(merged_df)}")
            
            # Merge aggregated order items
            merged_df = merged_df.merge(order_items_agg, on='order_id', how='left')
            logger.info(f"After order items merge: {len(merged_df)}")
            
            # Merge aggregated payments
            merged_df = merged_df.merge(payments_agg, on='order_id', how='left')
            logger.info(f"After payments merge: {len(merged_df)}")
            
            # Merge aggregated products
            merged_df = merged_df.merge(product_agg, on='order_id', how='left')
            logger.info(f"After products merge: {len(merged_df)}")
            
            # Merge customers
            merged_df = merged_df.merge(
                customers[['customer_id', 'customer_unique_id', 'customer_zip_code_prefix', 
                          'customer_city', 'customer_state']], 
                on='customer_id', 
                how='left'
            )
            logger.info(f"After customers merge: {len(merged_df)}")
            
            # FINAL CLEANUP: Drop rows with missing critical features
            critical_cols = ['payment_value', 'payment_installments', 'customer_state', 'review_score']
            merged_df.dropna(subset=critical_cols, inplace=True)
            logger.info(f"After final cleanup: {len(merged_df)}")

            # Generate health metrics
            nan_report = merged_df.isna().sum()
            nan_percent = (merged_df.isna().mean() * 100).round(2)
            
            logger.info("\nData Quality Report:")
            logger.info(f"Total rows: {len(merged_df):,}")
            logger.info(f"Total columns: {len(merged_df.columns)}")
            logger.info(f"Columns with missing values: {(nan_report > 0).sum()}")

            # Save validated data
            output_file = self.validated_data_dir / "merged_stage2.parquet"
            merged_df.to_parquet(output_file, index=False)
            logger.info(f"Saved validated data to: {output_file}")

            # Save validation report
            self.report["summary"] = {
                "total_rows": len(merged_df),
                "total_cols": len(merged_df.columns),
                "columns": list(merged_df.columns),
                "nan_counts": nan_report.to_dict(),
                "nan_percent": nan_percent.to_dict(),
                "output_file": str(output_file),
                "target_distribution": merged_df['review_score'].value_counts().to_dict()
            }
            save_json(Path(self.config.report_file), self.report)

            # Save status
            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Validation status: True\nRows: {len(merged_df)}\nColumns: {len(merged_df.columns)}")

            logger.info(f"✓ STAGE 2 COMPLETE. Final dataset: {len(merged_df):,} rows, {len(merged_df.columns)} columns")
            return True

        except Exception as e:
            logger.exception("Stage 2 failed")
            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Validation status: False\nError: {str(e)}")
            raise e
