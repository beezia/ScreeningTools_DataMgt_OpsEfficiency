import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import joblib
import time
import threading
import psutil
import onnxruntime as ort
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# -----------------------
# Load Models
# -----------------------
autoqc_model = joblib.load("models/autoqc_iso.pkl")

gpu_session = ort.InferenceSession(
    "models/hit_mlp.onnx",
    providers=["DmlExecutionProvider", "CPUExecutionProvider"]
)

cpu_session = ort.InferenceSession(
    "models/hit_mlp.onnx",
    providers=["CPUExecutionProvider"]
)

gpu_input_name = gpu_session.get_inputs()[0].name
cpu_input_name = cpu_session.get_inputs()[0].name

# -----------------------
# App Setup
# -----------------------
root = tk.Tk()
root.title("HTS Edge AI - Intelligent Screening Console")
root.geometry("1600x1000")

data = pd.DataFrame(columns=[
    "sample_id","assay_value","qc_metric",
    "instrument_temp","patient_delta",
    "anomaly_flag","hit_score","timestamp"
])

sample_counter = 0
batch_size = 200
start_time = time.time()

# -----------------------
# KPI BAR
# -----------------------
kpi_frame = ttk.Frame(root)
kpi_frame.pack(fill="x", pady=8)

total_label = ttk.Label(kpi_frame, text="Total Samples: 0", font=("Arial", 11))
total_label.pack(side="left", padx=8)

anomaly_label = ttk.Label(kpi_frame, text="Anomalies: 0", font=("Arial", 11))
anomaly_label.pack(side="left", padx=8)

top_hit_label = ttk.Label(kpi_frame, text="Top 1% Hits: 0", font=("Arial", 11))
top_hit_label.pack(side="left", padx=8)

samples_hr_label = ttk.Label(kpi_frame, text="Samples/hr: 0", font=("Arial", 11))
samples_hr_label.pack(side="left", padx=8)

cpu_util_label = ttk.Label(kpi_frame, text="CPU Utilization: 0%", font=("Arial", 11))
cpu_util_label.pack(side="left", padx=8)

igpu_util_label = ttk.Label(kpi_frame, text="iGPU Est. Utilization: 0%", font=("Arial", 11))
igpu_util_label.pack(side="left", padx=8)

cpu_latency_label = ttk.Label(kpi_frame, text="CPU Latency: 0ms", font=("Arial", 11))
cpu_latency_label.pack(side="left", padx=8)

igpu_latency_label = ttk.Label(kpi_frame, text="iGPU Latency: 0ms", font=("Arial", 11))
igpu_latency_label.pack(side="left", padx=8)

# -----------------------
# Matplotlib 2x3 Layout
# -----------------------
fig, ax = plt.subplots(2,3, figsize=(16,9))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill="both", expand=True)

# -----------------------
# Custom colormap for 96-well plate
# -----------------------
plate_cmap = ListedColormap(["red", "yellow", "green"])  # Low → Medium → High

# -----------------------
# Update Loop
# -----------------------
def update_dashboard():
    global data, sample_counter

    while True:
        # Simulate new batch of samples
        new_batch = pd.DataFrame({
            "sample_id": range(sample_counter+1, sample_counter+batch_size+1),
            "assay_value": np.random.normal(100,3,batch_size),
            "qc_metric": np.random.normal(0.0,1.0,batch_size),
            "instrument_temp": np.random.normal(37,0.5,batch_size),
            "patient_delta": np.random.normal(0,0.2,batch_size)
        })

        X = new_batch[["assay_value","qc_metric","instrument_temp","patient_delta"]]
        X_np = X.to_numpy(dtype=np.float32)

        # -----------------------
        # CPU Inference (AutoQC)
        # -----------------------
        start_cpu = time.time()
        cpu_session.run(None, {cpu_input_name: X_np})
        cpu_latency = (time.time() - start_cpu) * 1000  # in ms
        new_batch["anomaly_flag"] = autoqc_model.predict(X)

        # -----------------------
        # iGPU Inference (Hit Scoring)
        # -----------------------
        start_gpu = time.time()
        hit_scores = gpu_session.run(None, {gpu_input_name: X_np})[0].flatten()
        gpu_latency = (time.time() - start_gpu) * 1000
        new_batch["hit_score"] = hit_scores
        new_batch["timestamp"] = datetime.now()

        data = pd.concat([data, new_batch], ignore_index=True)
        sample_counter += batch_size

        # -----------------------
        # Metrics
        # -----------------------
        elapsed_hr = (time.time() - start_time)/3600
        samples_per_hr = int(sample_counter/elapsed_hr) if elapsed_hr > 0 else 0
        top_threshold = np.percentile(data["hit_score"], 99)
        data["top_hit"] = data["hit_score"] >= top_threshold
        anomalies = data[data["anomaly_flag"] == -1]

        total_label.config(text=f"Total Samples: {len(data)}")
        anomaly_label.config(text=f"Anomalies: {len(anomalies)}")
        top_hit_label.config(text=f"Top 1% Hits: {data['top_hit'].sum()}")
        samples_hr_label.config(text=f"Samples/hr: {samples_per_hr}")
        cpu_util_label.config(text=f"CPU Utilization: {psutil.cpu_percent()}%")
        igpu_util_label.config(text=f"iGPU Est. Utilization: {min(100, int(batch_size*0.2))}%")  # rough estimate
        cpu_latency_label.config(text=f"CPU Latency: {int(cpu_latency)}ms")
        igpu_latency_label.config(text=f"iGPU Latency: {int(gpu_latency)}ms")

        # -----------------------
        # TOP LEFT - Operational Stability
        # -----------------------
        ax[0,0].clear()
        ax[0,0].plot(data["assay_value"].tail(500))
        ax[0,0].set_title("Operational Stability - Instrument Health Monitor")
        ax[0,0].set_xlabel("Sample Index")
        ax[0,0].set_ylabel("Assay Value")
        ax[0,0].text(0.02, 0.95,
                     "• Is flat → stable instrument\n"
                     "• Slopes → calibration drift\n"
                     "• Becomes noisy → mechanical instability",
                     transform=ax[0,0].transAxes,
                     fontsize=8,
                     verticalalignment='top',
                     bbox=dict(boxstyle="round", alpha=0.2))

        # -----------------------
        # TOP MIDDLE - Biological Prioritization
        # -----------------------
        ax[0,1].clear()
        ax[0,1].scatter(
            data["hit_score"].tail(500),
            data["assay_value"].tail(500),
            c=["red" if a == -1 else "green" for a in data["anomaly_flag"].tail(500)],
            alpha=0.6
        )
        ax[0,1].set_title("Biological Prioritization: AI Ranking")
        ax[0,1].set_xlabel("AI Hit Score")
        ax[0,1].set_ylabel("Assay Value")
        top_hits = data[data["top_hit"]].tail(20)
        ax[0,1].scatter(top_hits["hit_score"], top_hits["assay_value"],
                        edgecolors="black", s=100, facecolors="none")
        for _, row in top_hits.iterrows():
            ax[0,1].annotate(f"ID {int(row['sample_id'])}",
                             (row["hit_score"], row["assay_value"]),
                             fontsize=7)
        ax[0,1].text(0.02, 0.95,
                     "• Black outlined = top 1% hits\n"
                     "• Labels = auto-identified best\n"
                     "• High hit score = high priority",
                     transform=ax[0,1].transAxes,
                     fontsize=8,
                     verticalalignment='top',
                     bbox=dict(boxstyle="round", alpha=0.2))

        # -----------------------
        # TOP RIGHT - 96-Well Plate Activity
        # -----------------------
        ax[0,2].clear()
        plate_data = np.random.randint(0,3,(8,12))
        ax[0,2].imshow(plate_data, cmap=plate_cmap, vmin=0, vmax=2)
        ax[0,2].set_title("96-Well Plate Activity")
        ax[0,2].set_xticks(range(12))
        ax[0,2].set_yticks(range(8))
        ax[0,2].set_xticklabels([str(i+1) for i in range(12)])
        ax[0,2].set_yticklabels([chr(65+i) for i in range(8)])
        legend_elements = [Patch(facecolor="green", edgecolor='k', label='High / Top Hits'),
                           Patch(facecolor="yellow", edgecolor='k', label='Medium Activity'),
                           Patch(facecolor="red", edgecolor='k', label='Low / Anomaly')]
        ax[0,2].legend(handles=legend_elements, loc="upper right", fontsize=7)

        # -----------------------
        # LOWER LEFT - Real-Time Anomaly Tracking
        # -----------------------
        ax[1,0].clear()
        ax[1,0].set_title("Real-Time Anomaly Tracking")
        ax[1,0].set_xlabel("Sample Index")
        ax[1,0].set_ylabel("Assay Value")
        if not anomalies.empty:
            ax[1,0].scatter(anomalies.index,
                            anomalies["assay_value"],
                            color="red",
                            s=20)
        ax[1,0].text(0.02, 0.95,
                     "• Red dot = operational anomalies",
                     transform=ax[1,0].transAxes,
                     fontsize=8,
                     verticalalignment='top',
                     bbox=dict(boxstyle="round", alpha=0.2))

        # -----------------------
        # LOWER MIDDLE - Workload Distribution
        # -----------------------
        ax[1,1].clear()
        ax[1,1].bar(["CPU","iGPU"], [psutil.cpu_percent(), min(100, int(batch_size*0.2))])
        ax[1,1].set_title("Workload Distribution")
        ax[1,1].set_ylabel("Utilization %")
        ax[1,1].set_ylim(0,100)
        ax[1,1].text(0.05,0.95,
                     "CPU          iGPU\n"
                     "AutoQC       Hit Scoring\n"
                     "Baseline inf Neural Network",
                     transform=ax[1,1].transAxes,
                     fontsize=8,
                     verticalalignment='top',
                     bbox=dict(boxstyle="round", alpha=0.2))

        # -----------------------
        # LOWER RIGHT - Workload Table
        # -----------------------
        ax[1,2].clear()
        ax[1,2].axis("off")
        table_data = [["CPU","AutoQC, Baseline inference"],
                      ["iGPU","Hit Scoring, Neural Network inference"]]
        table = ax[1,2].table(cellText=table_data,
                              colLabels=["Device","Workload"],
                              loc="center",
                              cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1,2)
        ax[1,2].set_title("Workload Mapping", fontsize=10)

        canvas.draw()
        time.sleep(3)

# -----------------------
# Start Background Thread
# -----------------------
thread = threading.Thread(target=update_dashboard, daemon=True)
thread.start()

root.mainloop()