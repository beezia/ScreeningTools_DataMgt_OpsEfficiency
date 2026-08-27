import pandas as pd
import lightgbm as lgb
import joblib
import os

os.makedirs("models", exist_ok=True)

data = pd.read_csv("hts_data.csv")
data["drift_target"] = data["instrument_temp"].shift(-1).fillna(method="ffill")

X = data[["instrument_temp"]]
y = data["drift_target"]

model = lgb.LGBMRegressor()
model.fit(X,y)

joblib.dump(model,"models/drift_model.pkl")
print("Drift model trained.")