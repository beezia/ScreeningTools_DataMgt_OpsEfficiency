import numpy as np
import pandas as pd

def generate_hts_data(n_samples=5000):
    np.random.seed(42)

    data = pd.DataFrame({
        "sample_id": np.arange(n_samples),
        "assay_value": np.random.normal(100, 10, n_samples),
        "qc_metric": np.random.normal(0, 1, n_samples),
        "instrument_temp": np.random.normal(37, 0.5, n_samples),
        "patient_delta": np.random.normal(0, 5, n_samples)
    })

    # Inject anomalies
    anomaly_idx = np.random.choice(n_samples, 50)
    data.loc[anomaly_idx, "assay_value"] += np.random.normal(50, 10, 50)

    data.to_csv("hts_data.csv", index=False)
    print("HTS synthetic data generated.")

if __name__ == "__main__":
    generate_hts_data()