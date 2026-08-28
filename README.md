# Screening Tools Edge AI Reference Application

## Overview

This reference application and demo showcases Screening Tools Edge AI running on a CPU and integrated GPU (iGPU). It demonstrates how edge AI can:

1. Detect anomalous results in real time with AutoQC.
2. Prioritize true biological hits over noise.
3. Reduce manual review workload.
4. Provide predictive operational insights.
5. Run entirely on commodity edge hardware.

## Problem Statement

High-throughput screening (HTS) tools, such as Abbott Alinity, Hologic Panther, and Siemens Atellica, generate thousands of assay results per hour. Although instrument automation has advanced, laboratories continue to face several challenges:

1. **Operational instability:** Subtle calibration drift or mechanical issues can compromise data quality.
2. **Biological prioritization:** Identifying top candidates quickly from thousands of samples is difficult without AI.
3. **Data overload:** Managing assay and quality-control metrics in real time is complex.
4. **Inefficient resource utilization:** CPU-heavy processing can leave iGPUs underutilized in edge deployments.

## Reference Solution

The reference solution runs an HTS Edge AI Console that combines AI-driven scoring, real-time quality-control monitoring, and workload-aware hardware acceleration. The demo runs on a standard Windows laptop with a CPU and iGPU.

## Getting Started

### Prerequisites

- Windows system with a CPU and integrated GPU
- Python
- Project dependencies listed in `requirements.txt`

### Setup and Execution

Open a terminal, change to the project folder, and run the following commands:

```powershell
cd <project-folder>
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Generate the simulated data, train the models, and run inference:

```powershell
python data_simulation.py
python autoqc_train.py
python autoqc_infer.py
python hit_train.py
python hit_infer.py
python drift_train.py
python drift_infer.py
```

Launch the dashboard:

```powershell
python ui_v3_ADLM.py
```

## User Interface

### Operational Stability Monitoring

The top-left quadrant provides continuous visualization of assay-value trends to detect calibration drift, noise, or instability.

- Flat trends indicate stable operation.
- Noisy trends can indicate potential mechanical issues.

### Biological Prioritization

The top-right quadrant uses a deep multilayer perceptron (MLP) model to rank samples by predicted biological relevance.

- Highlights the top 1% of hits with black-outlined markers.
- Labels the best candidates for rapid follow-up.
- Plots assay value against AI hit score for immediate interpretability.

### Real-Time Anomaly Tracking

The lower-left quadrant uses AutoQC to continuously flag operational anomalies.

- Red dots identify samples that require attention.
- The visualization provides immediate anomaly detection.

### Workload Distribution and Hardware Acceleration

The lower-right quadrant visualizes CPU and iGPU utilization.

- A dynamic bar chart shows CPU and iGPU utilization.
- A workload table identifies where AutoQC, hit scoring, baseline inference, and neural-network inference run.
- A batch-size slider can increase the iGPU workload to demonstrate hardware-acceleration efficiency.

### KPI and Latency Monitoring

The dashboard displays:

- Total samples
- Anomalies
- Top hits
- Throughput in samples per hour
- Real-time CPU latency
- Real-time iGPU latency

## What the Dashboard Simulates

### Total Samples

The total-samples metric represents the number of assay results processed in real time. Every few seconds, the application generates a new batch of synthetic samples to simulate incoming patient or screening samples.

### Anomalies: AutoQC Model

The Isolation Forest AutoQC model detects:

- Instrument drift
- Quality-control metric outliers
- Abnormal assay patterns
- Potential reagent or temperature instability

Red points on the scatter plot represent samples flagged as operational risks. In a real instrument, these signals could support preventive maintenance, prevent reporting of poor-quality results, reduce false positives and false negatives, and improve uptime.

### Average Hit Score: MLP Model Through ONNX

The hit-prioritization neural network runs on either the CPU or the iGPU through DirectML. It simulates AI scoring of compounds or samples based on the likelihood that they are true hits.

In high-throughput pharmaceutical screening, hit scoring can help:

- Reduce follow-up workload.
- Prioritize high-value samples.
- Improve downstream validation efficiency.

### Assay-Value Trend Chart

The top chart displays:

- Streaming assay values over time
- An operational-stability view
- Drift-detection visualization

In a laboratory environment, this view helps engineers monitor assay consistency.

### Hit Score Versus Assay Value

The scatter plot displays:

- **X-axis:** AI hit confidence
- **Y-axis:** Assay measurement
- **Red dots:** Anomalies

This view demonstrates how AI can score biological relevance and detect operational risk at the same time. Traditional HTS systems often handle these functions separately.

### CPU Versus iGPU Speedup

The application benchmarks:

- CPU inference time
- Intel Core Ultra iGPU inference time through DirectML

The comparison demonstrates edge AI acceleration without a discrete GPU. For companies building next-generation instruments, this can support:

- No additional discrete-GPU hardware cost
- No cloud dependency
- Real-time AI at the instrument
- Lower latency
- Lower bandwidth requirements
- Improved cybersecurity posture

## Application Flow

### AutoQC Models

**Scripts:** `autoqc_train.py` and `autoqc_infer.py`

**Purpose:** Detect operational anomalies in the screening instrument and provide real-time quality-control monitoring.

**Flow:**

1. `autoqc_train.py` trains the model on historical quality-control and instrument data.
2. `autoqc_infer.py` processes incoming sample data, including assay values and instrument readings, and predicts `anomaly_flag` values.

**Dashboard integration:**

- Feeds the real-time anomaly-tracking view in the lower-left quadrant.
- Displays red dots for flagged anomalies.
- Contributes to the **Anomalies** KPI.

**Hardware:** CPU, because AutoQC is lightweight.

**Latency:** CPU inference time can be measured per batch and displayed in the workload table.

### Drift Models

**Scripts:** `drift_train.py` and `drift_infer.py`

**Purpose:** Monitor long-term drift in assay values or instrument behavior.

**Flow:**

1. `drift_train.py` trains a deep MLP to detect trends or deviations over time.
2. `drift_infer.py` calculates drift scores for incoming batches.

**Dashboard integration:**

- Feeds the operational-stability chart in the top-left quadrant.
- Supports visualization of flat, sloped, or noisy trends.

**Hardware:** CPU, because the workload is lightweight.

**Latency:** Provides a fast baseline CPU workload.

### Hit-Scoring Models

**Scripts:** `hit_train.py` and `hit_infer.py`

**Purpose:** Rank samples by biological priority.

**Flow:**

1. `hit_train.py` trains a deep MLP on assay features to predict hit scores.
2. `hit_infer.py` generates hit scores for each incoming sample batch.

**Dashboard integration:**

- Feeds the biological-prioritization chart in the top-right quadrant.
- Highlights the top 1% of hits.
- Automatically labels top candidates.
- Contributes to the hit-scoring KPI.

**Hardware:** CPU or iGPU. The iGPU through DirectML is preferred for the demo.

**Latency:** Measures iGPU throughput for the **Samples/hour** metric and can be displayed in the workload table.

### User Interface

**Script:** `ui_v3_ADLM.py` or `ui.py`, depending on the project configuration.

**Purpose:** Connect the models and visualizations in a real-time dashboard.

**Flow:**

1. Generate a new sample batch containing `batch_size` rows.
2. Send the batch to AutoQC for anomaly prediction.
3. Send the batch to the hit-scoring model for biological-priority scoring.
4. Send the batch to the drift model for operational-stability analysis.
5. Update the KPI bar with total samples, anomalies, top hits, and samples per hour.
6. Update the workload table with CPU and iGPU tasks and optional latency measurements.
7. Update the CPU and iGPU utilization chart.

**Hardware visualization:** Displays CPU utilization, iGPU utilization, and workload mapping.

## Component Mapping

| Model or script | Input | Output | Hardware | UI component |
|---|---|---|---|---|
| `autoqc_train.py` | Historical quality-control data | Trained AutoQC model | CPU | Not applicable |
| `autoqc_infer.py` | New sample batch | `anomaly_flag` (`-1` or `1`) | CPU | Lower-left chart and KPI bar |
| `drift_train.py` | Historical assay and instrument data | Trained drift model | CPU | Not applicable |
| `drift_infer.py` | New sample batch | Drift score per sample | CPU | Top-left chart |
| `hit_train.py` | Sample features | Trained hit MLP model | CPU or iGPU | Not applicable |
| `hit_infer.py` | New sample batch | Hit score per sample | iGPU | Top-right chart and KPI bar |
| `ui.py` or `ui_v3_ADLM.py` | New sample batch (`X_new`) | Calls AutoQC, drift, and hit models | CPU and iGPU | All charts, KPI bar, and workload table |
