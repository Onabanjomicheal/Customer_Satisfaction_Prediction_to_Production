import joblib
import numpy as np
import pandas as pd
import os

# 1. Check if files exist before loading
model_path = 'artifacts/model_training/MLP.joblib'
transformer_path = 'artifacts/feature_transformation/transformer.pkl'

if not os.path.exists(model_path) or not os.path.exists(transformer_path):
    print("❌ Error: Artifacts not found. Run Stage 4 and 5 first.")
    exit()

# 2. Load artifacts
model = joblib.load(model_path)
preprocessor = joblib.load(transformer_path)

# 3. Define the features
num_features = [
    "price", "freight_value", "product_photos_qty", "product_weight_g",
    "product_length_cm", "product_height_cm", "product_width_cm",
    "payment_value", "payment_installments", "purchase_delivery_difference",
    "estimated_actual_delivery_difference", "purchase_delivery_diff_per_price"
]

# Extract names from the OneHotEncoder inside the preprocessor
cat_features = list(preprocessor.named_transformers_['cat'].get_feature_names_out())
all_features = num_features + cat_features + ["review_availability"]

# 4. Calculate Importance
# For MLP, we sum the absolute weights of the first hidden layer
importances = np.abs(model.coefs_[0]).sum(axis=1)

# 5. Create DataFrame
feat_imp = pd.DataFrame({'feature': all_features, 'importance': importances})
feat_imp = feat_imp.sort_values(by='importance', ascending=False)

print("\n" + "="*30)
print("TOP 5 FEATURES DRIVING PREDICTIONS")
print("="*30)
print(feat_imp.head(5))
print("="*30)