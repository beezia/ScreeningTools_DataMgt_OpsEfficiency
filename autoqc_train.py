import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

os.makedirs("models", exist_ok=True)

data = pd.read_csv("hts_data.csv")
X = data[["assay_value", "qc_metric", "instrument_temp", "patient_delta"]]

model = IsolationForest(contamination=0.01)
model.fit(X)

joblib.dump(model, "models/autoqc_iso.pkl")
print("AutoQC model trained.")