import joblib

# Load the transformer
transformer = joblib.load('artifacts/feature_transformation/transformer.pkl')

# 1. Check the Type
print(f"Transformer Type: {type(transformer)}")

# 2. If it's a ColumnTransformer, list the individual steps
if hasattr(transformer, 'transformers_'):
    print("\n--- Detected Transformers ---")
    for name, trans, cols in transformer.transformers_:
        print(f"Name: {name}")
        print(f"Columns affected: {cols}")
        print(f"Transformer logic: {trans}")
        print("-" * 30)
else:
    print("This is a simple transformer, not a ColumnTransformer.")