import streamlit as st
import requests
import os
from datetime import datetime

BACKEND_ENDPOINT = os.environ.get("BACKEND_SERVICE_URL", "http://localhost:8080/predict")

st.set_page_config(page_title="CS Intelligence Portal", layout="wide")
st.title("📊 Customer Experience Decision Support")

with st.form("main_form"):
    st.info("Input logistics and order data to assess customer sentiment risk.")
    c1, c2, c3 = st.columns(3)
    with c1:
        price = st.number_input("Unit Price", value=100.0)
        freight = st.number_input("Freight Cost", value=15.0)
        p_cat = st.selectbox("Category", ["beleza_saude", "perfumaria", "esporte_lazer"])
        state = st.selectbox("State", ["SP", "RJ", "MG", "BA"])
    with c2:
        val = st.number_input("Total Paid", value=115.0)
        p_type = st.selectbox("Payment", ["credit_card", "boleto", "voucher"])
        inst = st.slider("Installments", 1, 12, 1)
        photos = st.number_input("Photos", 1, 10, 1)
    with c3:
        d_days = st.number_input("Delivery Days", value=7.0)
        e_days = st.number_input("Est. Days", value=10.0)
        c_time = st.number_input("Carrier Lead Time", value=2.0)
    
    submit = st.form_submit_button("Analyze Customer Sentiment")

if submit:
    now = datetime.now()
    payload = {
        "carrier_handling_time": float(c_time), "delivery_time_days": float(d_days),
        "order_items_count": 1.0, "payment_value": float(val),
        "estimated_delivery_days": float(e_days), "avg_item_price": float(price),
        "product_photos_qty": float(photos), "is_weekend_order": 1 if now.weekday() >= 5 else 0,
        "order_hour": now.hour, "product_description_lenght": 500.0,
        "total_freight": float(freight), "total_price": float(price),
        "is_late_delivery": 1 if d_days > e_days else 0, "used_installments": float(inst),
        "payment_installments": int(inst), "order_month": now.month,
        "order_day_of_week": now.weekday(), "product_category_name": p_cat,
        "payment_type": p_type, "customer_state": state
    }

    try:
        response = requests.post(url=BACKEND_ENDPOINT, json=payload)
        if response.status_code == 200:
            res = response.json()
            meta = res['metadata']
            score = res['scores']['satisfaction_probability']
            
            st.divider()
            
            # Scenario Displays
            m1, m2, m3 = st.columns(3)
            m1.metric("Assessment", meta['interpretation'])
            m2.metric("Risk Level", meta['risk_level'])
            m3.metric("Satisfaction Score", f"{score:.2%}")

            if meta['alert_color'] == "GREEN":
                st.success(f"**Recommended Action:** {meta['recommended_action']}")
            elif meta['alert_color'] == "YELLOW":
                st.warning(f"**Recommended Action:** {meta['recommended_action']}")
            else:
                st.error(f"**Recommended Action:** {meta['recommended_action']}")
            
            st.progress(score)
        else:
            st.error(f"System Error: {response.text}")
    except Exception as e:
        st.error(f"Connection Failed: {e}")