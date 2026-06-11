import streamlit as st
import joblib
import numpy as np

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Fraud Guard AI", page_icon="🛡️")

# 2. LOAD ASSETS
try:
    model = joblib.load('../models/fraud_model.pkl')
    scaler = joblib.load('../models/scaler.pkl')
except:
    st.error("Error: Could not find model files.")
    st.stop()

# 3. HEADER & UI
st.title("🛡️ Credit Card Fraud Detection")
st.write("Enter the transaction details below. The model will analyze the patterns to determine risk.")
st.markdown("---")

# 4. INPUT SECTION
st.subheader("Transaction Features")
st.info("Tip: Fraudulent transactions usually have extreme negative values for V14, V12, and V10.")

col1, col2 = st.columns(2)

with col1:
    v14 = st.number_input("V14 (Highest Impact):", value=0.0, step=1.0)
    v12 = st.number_input("V12 (High Impact):", value=0.0, step=1.0)
    v17 = st.number_input("V17 (Medium Impact):", value=0.0, step=1.0)

with col2:
    v4 = st.number_input("V4 (High Impact):", value=0.0, step=1.0)
    v10 = st.number_input("V10 (High Impact):", value=0.0, step=1.0)
    amount = st.number_input("Transaction Amount ($):", value=0.0, step=10.0)

if st.button("Run Fraud Analysis"):
    input_data = np.zeros(29)
    input_data[13] = v14
    input_data[11] = v12
    input_data[16] = v17
    input_data[3] = v4
    input_data[9] = v10
    scaled_amount = scaler.transform([[amount]])
    input_data[28] = scaled_amount[0][0]

    prediction = model.predict([input_data])
    prediction_proba = model.predict_proba([input_data])

    st.markdown("---")
    if prediction[0] == 1:
        st.error(f"### 🚨 RESULT: HIGH RISK (FRAUD)")
        st.write(f"Confidence: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.success(f"### ✅ RESULT: LOW RISK (LEGITIMATE)")
        st.write(f"Confidence: {prediction_proba[0][0]*100:.2f}%")