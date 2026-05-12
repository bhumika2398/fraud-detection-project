import streamlit as st
import joblib
import numpy as np

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Fraud Guard AI", page_icon="🛡️")

# 2. LOAD ASSETS (The "Brain" and the "Tool")
# These files must be in the same folder as app.py
try:
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
except:
    st.error("Error: Could not find 'fraud_model.pkl' or 'scaler.pkl'.")

# 3. HEADER & UI
st.title("🛡️ Credit Card Fraud Detection")
st.write("Enter the transaction details below. The model will analyze the patterns to determine risk.")
st.markdown("---")

# 4. INPUT SECTION (Focusing on the top SHAP features)
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

 # --- UPDATE THIS PART IN YOUR app.py ---

if st.button("Run Fraud Analysis"):
    # 1. Create the blank slate of 29 features
    input_data = np.zeros(29)
    
    # 2. Fill in the values from your UI
    input_data[13] = v14
    input_data[11] = v12
    input_data[16] = v17
    input_data[3] = v4
    input_data[9] = v10
    
    # 3. SCALE ONLY THE AMOUNT (Index 28)
    # We take the user's amount, scale it, and put it in the last slot
    scaled_amount = scaler.transform([[amount]]) # The scaler only wants the amount
    input_data[28] = scaled_amount[0][0]
    
    # 4. Make the prediction using the full 29-feature array
    # We don't scale the whole thing, just the Amount part
    prediction = model.predict([input_data])
    prediction_proba = model.predict_proba([input_data])

    st.markdown("---")
    if prediction[0] == 1:
        st.error(f"### 🚨 RESULT: HIGH RISK (FRAUD)")
        st.write(f"Confidence: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.success(f"### ✅ RESULT: LOW RISK (LEGITIMATE)")
        st.write(f"Confidence: {prediction_proba[0][0]*100:.2f}%")