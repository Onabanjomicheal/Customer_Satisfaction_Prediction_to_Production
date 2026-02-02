import streamlit as st
import requests
import os

# --- 1. CONFIGURATION & STANDARDS ---
# When deployed on AWS, you will set BACKEND_SERVICE_URL to your App Runner URL.
# Example: https://xyz123.us-east-1.awsapprunner.com/predict
BACKEND_ENDPOINT = "http://44.220.246.255:8080/predict"
st.set_page_config(page_title="Customer Satisfaction Portal", layout="wide")

st.title("🛡️ Customer Satisfaction Prediction")
st.sidebar.info(f"Connected to: {BACKEND_ENDPOINT}")

# --- 2. INPUT UI ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        price = st.number_input("Price", value=100.0)
        p_cat = st.selectbox("Category", ["bed_bath_table", "health_beauty", "sports_leisure"])
        c_state = st.selectbox("Customer State", ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "GO", "ES", "PE"])
        s_state = st.selectbox("Seller State", ["SP", "RJ", "MG", "PR", "BA", "SC", "RS"])
        del_diff = st.number_input("Delivery Days", value=10.0)

    with col2:
        p_type = st.selectbox("Payment Type", ["credit_card", "boleto", "voucher", "debit_card"])
        p_val = st.number_input("Payment Value", value=100.0)
        installments = st.slider("Installments", 1, 24, 1)
        est_diff = st.slider("Estimated vs Actual Diff", -10, 10, 0)
        review_avail = st.selectbox("Review Available", [1, 0])

    submit = st.form_submit_button("Run Inference")

# --- 3. INFERENCE LOGIC ---
if submit:
    # Constructing the payload exactly as your FastAPI schema expects
    payload = {
        "price": price, 
        "freight_value": 15.0, 
        "product_category_name": p_cat,
        "product_name_lenght": 40.0, 
        "product_description_lenght": 500.0,
        "product_photos_qty": 1.0, 
        "product_weight_g": 1000.0,
        "product_length_cm": 20.0, 
        "product_height_cm": 10.0, 
        "product_width_cm": 15.0,
        "seller_state": s_state, 
        "customer_state": c_state,
        "payment_sequential": 1.0, 
        "payment_type": p_type,
        "payment_installments": float(installments), 
        "payment_value": p_val,
        "review_availability": int(review_avail), 
        "purchase_delivery_difference": float(del_diff),
        "estimated_actual_delivery_difference": float(est_diff),
        "price_category": "affordable", 
        "purchase_delivery_diff_per_price": float(del_diff/price) if price != 0 else 0.0
    }

    try:
        with st.spinner("Wait for it..."):
            response = requests.post(
                url=BACKEND_ENDPOINT, 
                json=payload,
                timeout=15 
            )
        
        if response.status_code == 200:
            res_json = response.json()
            # Assuming your FastAPI returns {"prediction": 4}
            prediction = res_json.get("prediction", "Unknown")
            st.success(f"Model Prediction (Satisfaction Score): **{prediction}**")
        else:
            st.error(f"Backend Error: {response.status_code} - {response.text}")

    except requests.exceptions.ConnectionError:
        st.error(f"Failed to connect to the backend at {BACKEND_ENDPOINT}. Is the service live?")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")