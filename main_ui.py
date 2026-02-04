import streamlit as st
import requests
import os
import logging

# --- 1. PRODUCTION CONFIGURATION (Standard ML Pattern) ---
# We use os.environ.get to ensure the app is portable across Dev, Staging, and Prod.
BACKEND_ENDPOINT = os.environ.get("BACKEND_SERVICE_URL", "http://localhost:8080/predict")

# Initialize logging for production audit trails
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Customer Satisfaction Portal | MLOps Prod", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ Customer Satisfaction Prediction")
st.sidebar.header("System Status")
st.sidebar.info(f"Production API: {BACKEND_ENDPOINT}")

# --- 2. FEATURE INPUTS ---
with st.form("prediction_form"):
    st.subheader("Order & Delivery Details")
    col1, col2 = st.columns(2)
    
    with col1:
        price = st.number_input("Price ($)", min_value=0.0, value=100.0)
        p_cat = st.selectbox("Product Category", ["bed_bath_table", "health_beauty", "sports_leisure"])
        c_state = st.selectbox("Customer State", ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "GO", "ES", "PE"])
        s_state = st.selectbox("Seller State", ["SP", "RJ", "MG", "PR", "BA", "SC", "RS"])
        del_diff = st.number_input("Actual Delivery Days", min_value=0.0, value=10.0)

    with col2:
        p_type = st.selectbox("Payment Method", ["credit_card", "boleto", "voucher", "debit_card"])
        p_val = st.number_input("Total Payment Value", min_value=0.0, value=100.0)
        installments = st.slider("Installments", 1, 24, 1)
        est_diff = st.slider("Days vs Estimate (Negative = Early)", -15, 15, 0)
        review_avail = st.radio("Is Review Data Available?", [1, 0], horizontal=True)

    submit = st.form_submit_button("🚀 Run Production Inference")

# --- 3. PRODUCTION INFERENCE LOGIC ---
if submit:
    # Standardizing Payload to match ML Model Schema
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
        with st.spinner("Querying ML Model..."):
            response = requests.post(
                url=BACKEND_ENDPOINT, 
                json=payload,
                timeout=20 # Standard production timeout
            )
        
        if response.status_code == 200:
            prediction = response.json().get("prediction", "N/A")
            st.success(f"### Predicted Satisfaction Score: **{prediction}**")
            st.balloons()
            logger.info(f"Successful prediction: {prediction}")
        else:
            st.error(f"Upstream Service Error: {response.status_code}")
            logger.error(f"Backend returned error: {response.text}")

    except requests.exceptions.RequestException as e:
        st.error("Network Latency or Connection Error. Check Production Logs.")
        logger.error(f"Connection Failed: {e}")