import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection AI", page_icon="💳", layout="wide")

# ---------------- LOGIN SYSTEM ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login to Fraud Detection System")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💳 AI Powered Credit Card Fraud Detection System")

# ---------------- SIDEBAR ----------------

menu = st.sidebar.radio("Navigation", ["Single Prediction", "Bulk Prediction (CSV)"])

# ---------------- SINGLE PREDICTION ----------------
if menu == "Single Prediction":

    st.subheader("Enter Transaction Details")

    feature_names = [
        "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
        "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
        "V21","V22","V23","V24","V25","V26","V27","V28","Amount"
    ]

    cols = st.columns(3)
    input_data = []

    for i, feature in enumerate(feature_names):
        value = cols[i % 3].number_input(feature, value=0.0)
        input_data.append(value)

    if st.button("Analyze Transaction"):
        input_array = np.array([input_data])
        scaled_data = scaler.transform(input_array)

        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)

        fraud_prob = probability[0][1] * 100

        if prediction[0] == 1:
            st.error(f"⚠ Fraudulent Transaction Detected! ({fraud_prob:.2f}% confidence)")
        else:
            st.success(f"✅ Genuine Transaction ({100 - fraud_prob:.2f}% confidence)")

# ---------------- BULK CSV PREDICTION ----------------
elif menu == "Bulk Prediction (CSV)":

    st.subheader("Upload CSV File")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("Preview of Uploaded Data")
        st.dataframe(df.head())

        scaled = scaler.transform(df)
        predictions = model.predict(scaled)

        df["Prediction"] = predictions

        st.write("Prediction Results")
        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results CSV", csv, "fraud_predictions.csv")

# ---------------- MODEL EVALUATION ----------------


   

# streamlit run app.py
# & "C:\Users\Dinesh Pandey\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run app.py
