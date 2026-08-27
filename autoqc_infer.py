import pandas as pd
import joblib

data = pd.read_csv("hts_data.csv")
model = joblib.load("models/autoqc_iso.pkl")

X = data[["assay_value", "qc_metric", "instrument_temp", "patient_delta"]]
data["anomaly_flag"] = model.predict(X)

print(data[data["anomaly_flag"] == -1].head())