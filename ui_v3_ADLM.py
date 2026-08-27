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
from PIL import Image, ImageTk

# -----------------------
# Load Models
# -----------------------
autoqc_model = joblib.load("models/autoqc_iso.pkl")

gpu_session = ort.InferenceSession(
    "models/hit_mlp.onnx",
    providers=["DmlExecutionProvider", "CPUExecutionProvider"]
)

gpu_input_name = gpu_session.get_inputs()[0].name

# NOTE: cpu_session is removed in this version.
# The CPU workload is now genuinely distinct: it runs the sklearn AutoQC model.
# The iGPU workload is the ONNX neural net hit scorer.
# This gives the bar chart two authentically different latency sources.

# -----------------------
# App Setup
# -----------------------
root = tk.Tk()
root.title("HTS Edge AI - Intelligent Screening Console")
root.geometry("1600x1050")

data = pd.DataFrame(columns=[
    "sample_id", "assay_value", "qc_metric",
    "instrument_temp", "patient_delta",
    "anomaly_flag", "hit_score", "timestamp"
])

sample_counter = 0
batch_size = 200
start_time = time.time()

# Track last latencies for workload bar chart (used for % split)
last_cpu_latency = 0.0
last_gpu_latency = 0.0

# -----------------------
# Demo Title
# -----------------------
title_label = ttk.Label(root, text="Screening Tools Edge AI: Data Management & Operational Efficiency",
                        font=("Arial", 16, "bold"))
title_label.pack(pady=(2,1))

# -----------------------
# Top Frame: KPI + Logo
# -----------------------
top_frame = ttk.Frame(root)
top_frame.pack(fill="x", pady=(2,1))

total_label = ttk.Label(top_frame, text="Total Samples: 0", font=("Arial", 11))
total_label.pack(side="left", padx=8)

anomaly_label = ttk.Label(top_frame, text="Anomalies: 0", font=("Arial", 11))
anomaly_label.pack(side="left", padx=8)

top_hit_label = ttk.Label(top_frame, text="Top 1% Hits: 0", font=("Arial", 11))
top_hit_label.pack(side="left", padx=8)

samples_hr_label = ttk.Label(top_frame, text="Samples/hr: 0", font=("Arial", 11))
samples_hr_label.pack(side="left", padx=8)

cpu_util_label = ttk.Label(top_frame, text="CPU Latency (AutoQC): 0 ms", font=("Arial", 11))
cpu_util_label.pack(side="left", padx=8)

gpu_util_label = ttk.Label(top_frame, text="iGPU Latency (Hit Score): 0 ms", font=("Arial", 11))
gpu_util_label.pack(side="left", padx=8)

# Logo
logo_img = Image.open("images/logo.png").resize((140, 100))
logo_photo = ImageTk.PhotoImage(logo_img)
logo_label = tk.Label(top_frame, image=logo_photo)
logo_label.pack(side="right", padx=10)

# -----------------------
# Data Flow Pipeline
# -----------------------
pipeline_frame = ttk.Frame(root)
pipeline_frame.pack(fill="x", padx=10, pady=(0,2))

pipeline_label = ttk.Label(pipeline_frame,
                           text="Sample → Assay → [CPU] AutoQC → [iGPU] Hit Scoring → Prioritization → Review",
                           font=("Arial", 11))
pipeline_label.pack(side="left")

# -----------------------
# Workload Table
# -----------------------
table_frame = ttk.Frame(root)
table_frame.pack(fill="x", padx=10, pady=(0,2))
columns = ("Hardware", "Workload", "Latency (ms)", "Notes")
workload_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=3)
for col in columns:
    workload_table.heading(col, text=col)
    workload_table.column(col, width=200, anchor="center")
workload_table.pack()

# -----------------------
# Matplotlib 4-Quadrant Layout
# -----------------------
#fig, ax = plt.subplots(2, 2, figsize=(13, 8))
fig, ax = plt.subplots(
    2,
    2,
    figsize=(13,8),
    constrained_layout=True
)
canvas = FigureCanvasTkAgg(fig, master=root)
#canvas.get_tk_widget().pack(fill="both", expand=True)
canvas.get_tk_widget().pack(
    fill="both",
    expand=True,
    pady=(0,0)
)

# -----------------------
# Update Loop
# -----------------------
def update_dashboard():
    global data, sample_counter, last_cpu_latency, last_gpu_latency

    while True:
        # --- Generate dummy batch data ---
        new_batch = pd.DataFrame({
            "sample_id": range(sample_counter + 1, sample_counter + batch_size + 1),
            "assay_value": np.random.normal(100, 3, batch_size),
            "qc_metric": np.random.normal(0.0, 1.0, batch_size),
            "instrument_temp": np.random.normal(37, 0.5, batch_size),
            "patient_delta": np.random.normal(0, 0.2, batch_size)
        })

        X_np = new_batch[["assay_value", "qc_metric", "instrument_temp", "patient_delta"]].to_numpy(dtype=np.float32)

        # -------------------------------------------------------
        # OPTION 4: Distinct workloads per hardware unit
        # CPU  → sklearn IsolationForest AutoQC (genuine CPU-only work)
        # iGPU → ONNX neural net hit scoring   (DML-accelerated)
        # -------------------------------------------------------

        # CPU workload: AutoQC anomaly detection
        cpu_start = time.time()
        anomaly_flags = autoqc_model.predict(X_np)
        cpu_latency = (time.time() - cpu_start) * 1000  # ms

        # iGPU workload: neural net hit scoring
        gpu_start = time.time()
        hit_scores = gpu_session.run(None, {gpu_input_name: X_np})[0].flatten()
        gpu_latency = (time.time() - gpu_start) * 1000  # ms

        last_cpu_latency = cpu_latency
        last_gpu_latency = gpu_latency

        new_batch["anomaly_flag"] = anomaly_flags
        new_batch["hit_score"] = hit_scores
        new_batch["timestamp"] = datetime.now()

        data = pd.concat([data, new_batch], ignore_index=True)
        sample_counter += batch_size

        # --- KPI calculations ---
        elapsed_hr = (time.time() - start_time) / 3600
        samples_per_hr = int(sample_counter / elapsed_hr) if elapsed_hr > 0 else 0
        top_threshold = np.percentile(data["hit_score"], 99)
        data["top_hit"] = data["hit_score"] >= top_threshold
        anomalies = data[data["anomaly_flag"] == -1]

        total_label.config(text=f"Total Samples: {len(data)}")
        anomaly_label.config(text=f"Anomalies: {len(anomalies)}")
        top_hit_label.config(text=f"Top 1% Hits: {data['top_hit'].sum()}")
        samples_hr_label.config(text=f"Samples/hr: {samples_per_hr}")
        cpu_util_label.config(text=f"CPU Latency (AutoQC): {cpu_latency:.1f} ms")
        gpu_util_label.config(text=f"iGPU Latency (Hit Score): {gpu_latency:.1f} ms")

        # --- Workload table: now clearly shows what each unit is doing ---
        workload_table.delete(*workload_table.get_children())
        workload_table.insert("", "end", values=("CPU", "AutoQC (IsolationForest)", f"{cpu_latency:.1f}", "sklearn inference"))
        workload_table.insert("", "end", values=("iGPU", "Hit Scoring (MLP)", f"{gpu_latency:.1f}", "ONNX / DirectML"))
        workload_table.insert("", "end", values=("CPU + iGPU", "Total Pipeline", f"{cpu_latency + gpu_latency:.1f}", "parallel"))

        # --- Derive % split from actual measured latencies ---
        total_latency = cpu_latency + gpu_latency
        cpu_pct = (cpu_latency / total_latency) * 100 if total_latency > 0 else 50
        gpu_pct = (gpu_latency / total_latency) * 100 if total_latency > 0 else 50

        # -----------------------
        # TOP LEFT - Operational Stability
        # -----------------------
        ax[0, 0].clear()
        ax[0, 0].plot(data["assay_value"].tail(500))
        ax[0, 0].set_title("Operational Stability - Instrument Health Monitor")
        ax[0, 0].set_xlabel("Sample Index")
        ax[0, 0].set_ylabel("Assay Value")

        # -----------------------
        # TOP RIGHT - Biological Prioritization
        # -----------------------
        ax[0, 1].clear()
        ax[0, 1].scatter(
            data["hit_score"].tail(500),
            data["assay_value"].tail(500),
            c=["red" if a == -1 else "green" for a in data["anomaly_flag"].tail(500)],
            alpha=0.6
        )
        top_hits = data[data["top_hit"]].tail(20)
        ax[0, 1].scatter(top_hits["hit_score"], top_hits["assay_value"],
                         edgecolors="black", s=100, facecolors="none")
        for _, row in top_hits.iterrows():
            ax[0, 1].annotate(f"ID {int(row['sample_id'])}",
                              (row["hit_score"], row["assay_value"]), fontsize=7)
        ax[0, 1].set_title("Biological Prioritization: AI Ranking of Biologically Meaningful Samples")
        ax[0, 1].set_xlabel("AI Hit Score")
        ax[0, 1].set_ylabel("Assay Value (Biological Signal)")

        # -----------------------
        # LOWER LEFT - Real-Time Anomaly Tracking
        # -----------------------
        ax[1, 0].clear()
        ax[1, 0].set_title("Real-Time Anomaly Tracking: Continuous AI-Driven QC Monitoring")
        ax[1, 0].set_xlabel("Sample Index")
        ax[1, 0].set_ylabel("Assay Value")
        if not anomalies.empty:
            ax[1, 0].scatter(anomalies.index, anomalies["assay_value"], color="red", s=20)

        # -----------------------
        # BOTTOM RIGHT - Workload Distribution (Option 4 applied)
        # Split is derived from the two genuinely separate workload latencies.
        # CPU does AutoQC; iGPU does hit scoring — no redundant dual inference.
        # -----------------------
        ax[1, 1].clear()
        bars = ax[1, 1].bar(
            ["CPU\n(AutoQC)", "iGPU\n(Hit Scoring)"],
            [cpu_pct, gpu_pct],
            color=["steelblue", "darkorange"]
        )
        # Annotate bars with latency ms and derived %
        labels = [
            f"{cpu_pct:.1f}%\n({cpu_latency:.1f} ms)",
            f"{gpu_pct:.1f}%\n({gpu_latency:.1f} ms)"
        ]
        for bar, label in zip(bars, labels):
            ax[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                label,
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )
        ax[1, 1].set_ylim(0, 110)
        ax[1, 1].set_title(
            "Workload Distribution: CPU vs iGPU\n"
            "CPU: AutoQC (IsolationForest) | iGPU: Hit Scoring (MLP/ONNX)"
        )
        ax[1, 1].set_ylabel("Share of Inference Time %")

        canvas.draw()
        time.sleep(3)


# -----------------------
# Start Background Thread
# -----------------------
thread = threading.Thread(target=update_dashboard, daemon=True)
thread.start()

root.mainloop()
