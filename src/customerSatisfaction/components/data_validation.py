import os
import pandas as pd
from pathlib import Path
from customerSatisfaction.entity.config_entity import DataValidationConfig
from customerSatisfaction.utils.common import save_json
from customerSatisfaction import logger

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.schema = config.all_schema # Holds the nested schema dictionary
        self.report = {}
        self.raw_data_dir = Path(config.unzip_data_dir)
        self.validated_data_dir = Path(config.raw_validated_dir)
        self.validated_data_dir.mkdir(parents=True, exist_ok=True)

    def initiate_data_validation(self) -> bool:
        try:
            # Task 1: Loading & Schema Validation
            logger.info("Task 1: Loading datasets and verifying nested schema...")
            datasets = {}
            for file_name in self.config.datasets:
                path = self.raw_data_dir / file_name
                df = pd.read_csv(path)
                
                # Check if file exists in schema.yaml before validating
                if file_name in self.schema:
                    expected_cols = list(self.schema[file_name]["columns"].keys())
                    actual_cols = list(df.columns)
                    
                    # Validate columns
                    for col in expected_cols:
                        if col not in actual_cols:
                            logger.error(f"Missing column '{col}' in {file_name}")
                            return False
                
                datasets[file_name.replace(".csv", "")] = df
            
            cust = datasets["olist_customers_dataset"]
            orders = datasets["olist_orders_dataset"]
            items = datasets["olist_order_items_dataset"]
            pay = datasets["olist_order_payments_dataset"]
            prod = datasets["olist_products_dataset"]
            rev = datasets["olist_order_reviews_dataset"]
            sell = datasets["olist_sellers_dataset"]
            logger.info("Task 1 Completed: All datasets loaded and schema verified.")

            # Task 2: Deduplication
            logger.info("Task 2: Applying deduplication rules...")
            orders.drop_duplicates(subset=["order_id"], inplace=True)
            cust.drop_duplicates(subset=["customer_id"], inplace=True)
            prod.drop_duplicates(subset=["product_id"], inplace=True)
            sell.drop_duplicates(subset=["seller_id"], inplace=True)
            items.drop_duplicates(subset=["order_id", "product_id", "seller_id"], inplace=True)
            pay.drop_duplicates(subset=["order_id", "payment_type"], inplace=True)
            rev.drop_duplicates(subset=["order_id"], inplace=True)
            logger.info("Task 2 Completed: Deduplication finished.")

            # Task 3: Datetime Parsing
            logger.info("Task 3: Parsing datetime columns and removing NaT rows...")
            datetime_cols = [
                "order_purchase_timestamp", "order_approved_at",
                "order_delivered_carrier_date", "order_delivered_customer_date",
                "order_estimated_delivery_date"
            ]
            for col in datetime_cols:
                if col in orders.columns:
                    orders[col] = pd.to_datetime(orders[col], errors="coerce")
            
            orders.dropna(subset=datetime_cols, inplace=True)
            logger.info("Task 3 Completed: Datetime processing and NaT removal finished.")

            # Task 4: Merging
            logger.info("Task 4: Executing relational merge...")
            merged_df = (
                items
                .merge(prod, on="product_id", how="left")
                .merge(sell, on="seller_id", how="left")
                .merge(orders, on="order_id", how="left")
                .merge(cust, on="customer_id", how="left")
                .merge(rev, on="order_id", how="left")
                .merge(pay, on="order_id", how="left")
            )
            logger.info(f"Task 4 Completed: Merge finished. Table shape: {merged_df.shape}")

            # Task 5: Review Availability Flag
            logger.info("Task 5: Creating review availability flags...")
            merged_df["review_availability"] = (~merged_df["review_comment_message"].isna()).astype(int)
            
            # Task 6: Imputation
            logger.info("Task 6: Imputing missing text...")
            merged_df["review_comment_message"] = merged_df["review_comment_message"].fillna("indisponível")
            if "review_title" in merged_df.columns:
                merged_df["review_title"] = merged_df["review_title"].fillna("indisponível")
            logger.info("Task 6 Completed: Text imputation finished.")

            # Task 7: Benchmarking & Health Check
            logger.info("Task 7: Benchmarking Dataset Health...")
            nan_report = merged_df.isna().sum()
            nan_percent = (merged_df.isna().mean() * 100).round(2)
            
            print("\n" + "="*50)
            print("DATASET HEALTH REPORT (TOP 10 NaNs)")
            print("="*50)
            print(pd.concat([nan_report, nan_percent], axis=1, keys=['Count', '%']).sort_values(by='%', ascending=False).head(10))
            print("="*50 + "\n")

            # Task 8: Export
            logger.info("Task 8: Exporting clean data to Parquet format...")
            output_file = self.validated_data_dir / "merged_stage2.parquet"
            merged_df.to_parquet(output_file, index=False)
            logger.info(f"Task 8 Completed: Data saved to {output_file}")
            
            # Task 9: Finalizing Report
            self.report["summary"] = {
                "total_rows": len(merged_df),
                "total_cols": len(merged_df.columns),
                "nan_counts": nan_report.to_dict(),
                "output_file": str(output_file)
            }
            save_json(Path(self.config.report_file), self.report)

            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Validation status: True\nRows: {len(merged_df)}")

            logger.info(">>>>>> DATA VALIDATION STAGE COMPLETED <<<<<<")
            return True

        except Exception as e:
            logger.exception("Stage 2 Data Validation/Merge failed")
            with open(self.config.STATUS_FILE, "w") as f:
                f.write("Validation status: False")
            raise e