import streamlit as st
import pandas as pd
import joblib

st.title("AI-Enabled HTS Edge Node")

data = pd.read_csv("hts_data.csv")

model = joblib.load("models/autoqc_iso.pkl")
X = data[["assay_value","qc_metric","instrument_temp","patient_delta"]]
data["anomaly_flag"] = model.predict(X)

st.subheader("Anomaly Detection")
st.write(data[data["anomaly_flag"] == -1])

st.subheader("Assay Distribution")
st.line_chart(data["assay_value"])