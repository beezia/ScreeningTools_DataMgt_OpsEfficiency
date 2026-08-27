# dashboard_live.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import onnxruntime as ort
import matplotlib.pyplot as plt

st.set_page_config(page_title="HTS Live AI Dashboard", layout="wide")
st.title("🧬 HTS Edge AI - Live Streaming Dashboard")

# --- Load Models ---
autoqc_model = joblib.load("models/autoqc_iso.pkl")
hit_session = ort.InferenceSession("models/hit_mlp.onnx",
                                   providers=["DmlExecutionProvider", "CPUExecutionProvider"])
hit_input_name = hit_session.get_inputs()[0].name

# --- Sidebar ---
st.sidebar.header("Live Dashboard Settings")
batch_size = st.sidebar.slider("Batch size per update", min_value=50, max_value=1000, value=200)
update_interval = st.sidebar.slider("Update interval (seconds)", min_value=1, max_value=10, value=3)

# --- Initialize empty dataframe ---
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=[
        "sample_id","assay_value","qc_metric",
        "instrument_temp","patient_delta",
        "anomaly_flag","anomaly_flag_bool","hit_score"
    ])

# --- Placeholder containers ---
kpi_container = st.container()
chart_container = st.container()
table_container = st.container()
benchmark_container = st.container()

sample_counter = len(st.session_state.data)

# --- Main live loop ---
while True:
    # --- Simulate new batch of samples ---
    new_batch = pd.DataFrame({
        "sample_id": range(sample_counter+1, sample_counter+batch_size+1),
        "assay_value": np.random.normal(100,15,batch_size),
        "qc_metric": np.random.normal(0.0,1.0,batch_size),
        "instrument_temp": np.random.normal(37,0.5,batch_size),
        "patient_delta": np.random.normal(0,0.2,batch_size)
    })

    # --- AutoQC anomaly detection ---
    X_new = new_batch[["assay_value","qc_metric","instrument_temp","patient_delta"]]
    new_batch["anomaly_flag"] = autoqc_model.predict(X_new)
    new_batch["anomaly_flag_bool"] = new_batch["anomaly_flag"] == -1

    # --- Hit prioritization ---
    X_np = X_new.to_numpy(dtype=np.float32)
    hit_scores = hit_session.run(None, [X_np])[0].flatten()
    new_batch["hit_score"] = hit_scores

    # --- Append to session state ---
    st.session_state.data = pd.concat([st.session_state.data, new_batch], ignore_index=True)
    sample_counter += batch_size

    # --- Update KPIs ---
    with kpi_container:
        st.subheader("📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", len(st.session_state.data))
        col2.metric("Anomalies", st.session_state.data["anomaly_flag_bool"].sum())
        col3.metric("Avg Assay Value", f"{st.session_state.data['assay_value'].mean():.2f}")
        col4.metric("Avg Hit Score", f"{st.session_state.data['hit_score'].mean():.2f}")

    # --- Update Charts ---
    with chart_container:
        st.subheader("📈 Assay Value Trend")
        st.line_chart(st.session_state.data.set_index("sample_id")["assay_value"])

        st.subheader("🔴 Hit Score vs Assay Value")
        fig, ax = plt.subplots(figsize=(8,4))
        colors = np.where(st.session_state.data["anomaly_flag_bool"], 'red', 'green')
        ax.scatter(st.session_state.data["hit_score"], st.session_state.data["assay_value"], c=colors, alpha=0.6)
        ax.set_xlabel("Hit Score")
        ax.set_ylabel("Assay Value")
        st.pyplot(fig)

    # --- Update Top Anomalies Table ---
    with table_container:
        st.subheader("Top 10 Anomalies")
        top_anomalies = st.session_state.data[st.session_state.data["anomaly_flag_bool"]].nlargest(10,"assay_value")
        st.dataframe(top_anomalies[[
            "sample_id","assay_value","qc_metric","instrument_temp","patient_delta","hit_score"
        ]])

    # --- Benchmark iGPU vs CPU for recent batch ---
    with benchmark_container:
        st.subheader("⚡ Inference Benchmark (Recent Batch)")
        # CPU
        cpu_session = ort.InferenceSession("models/hit_mlp.onnx", providers=["CPUExecutionProvider"])
        input_name_cpu = cpu_session.get_inputs()[0].name
        start = time.time()
        cpu_session.run(None, [X_np])
        cpu_time = time.time() - start
        # iGPU
        gpu_session = ort.InferenceSession("models/hit_mlp.onnx", providers=["DmlExecutionProvider"])
        input_name_gpu = gpu_session.get_inputs()[0].name
        start = time.time()
        gpu_session.run(None, [X_np])
        gpu_time = time.time() - start

        col1, col2, col3 = st.columns(3)
        col1.metric("Batch Size", X_np.shape[0])
        col2.metric("CPU Time (s)", f"{cpu_time:.4f}")
        col3.metric("iGPU Time (s)", f"{gpu_time:.4f}")
        st.markdown(f"**Speedup:** {cpu_time/gpu_time:.2f}x")

    time.sleep(update_interval)