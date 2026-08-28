# ScreeningTools_DataMgt_OpsEfficiency

**Overview**
<br><br>
This is a reference application and a demo showcasing Screening Tools Edge AI running on CPU + integrated GPU, that can:<br>
	1. Detect anomalous results in real time (Auto-QC)<br>
	2. Prioritize true biological “hits” vs noise <br>
	3. Reduce manual review workload <br>
	4. Provide predictive operational insights <br>
    5. Run entirely on commodity edge hardware <br>

<br>
High-throughput screening (HTS) tools like Abbott Alinity, Hologic Panther, and Siemens Atellica—generate thousands of assay results per hour. While instrument automation has advanced, laboratories face persistent challenges:<br>
	1. Operational instability: subtle calibration drifts or mechanical issues can compromise data quality.<br>
	2. Biological prioritization: identifying top candidates quickly from thousands of samples is slow without AI.<br>
	3. Data overload: managing assay and QC metrics in real time is complex.<br>
	4. Inefficient resource usage: CPU-heavy processing leaves iGPUs underutilized in edge deployments.<br>

<br>
This reference Solution (can be a demo)
This reference solution runs an HTS Edge AI Console that addresses these challenges by combining AI-driven scoring, real-time QC monitoring, and workload-aware hardware acceleration on a standard Windows laptop with CPU + iGPU.<br>

**Steps to run the reference solution**: 

cd into the folder <br>
python -m venv venv  # start virtual environment <br>
venv\Scripts\activate <br>
pip install -r requirements.txt <br>
 
python data_simulation.py <br>
python autoqc_train.py <br>
python autoqc_infer.py <br>
python hit_train.py <br>
python hit_infer.py <br>
python drift_train.py <br>
python drift_infer.py <br>

python ui_v3_ADLM.py <br>

<br><br>

**What does the User Interface (UI) shows**: 
1.	Operational Stability Monitoring (Top Left Quadrant):
      o	Continuous visualization of assay value trends to detect calibration drift, noise, or instability.
      o	Provides immediate visual cues: flat trends → stable, noisy trends → potential mechanical issues.
2.	Biological Prioritization (Top Right Quadrant):
      o	Deep MLP AI model ranks samples based on predicted biological relevance.
      o	Highlights top 1% hits with black-outlined markers and labels best candidates for rapid follow-up.
      o	Scatter plot shows assay value vs AI hit score for instant interpretability.
3.	Real-Time Anomaly Tracking (Lower Left Quadrant):
      o	AutoQC continuously flags operational anomalies using AI.
      o	Red dots indicate samples requiring attention, providing instant anomaly detection.
4.	Workload Distribution & Hardware Acceleration (Lower Right Quadrant):
      o	Dynamic bar chart shows CPU vs iGPU utilization.
      o	Table indicates which workloads run where: AutoQC, Hit Scoring, Baseline inference, Neural network inference.
      o	Batch size slider allows stressing the iGPU to demonstrate hardware acceleration efficiency.
5.	KPI Metrics & Latency Monitoring:
      o	Total samples, anomalies, top hits, throughput (samples/hr).
      o	Real-time CPU and iGPU latency displayed to demonstrate hardware performance.

**What the dashboard is simulating**: 
Total Samples
This simulates: Number of assay results processed in real time.
Every few seconds, a new batch of synthetic samples is generated (like incoming patient or screening samples).

Anomalies (AutoQC Model)
This comes from your Isolation Forest AutoQC model.
It detects:
	• Instrument drift
	• QC metric outliers
	• Abnormal assay patterns
	• Potential reagent or temperature instability
Red points on the scatter plot = flagged as operational risk.
  In a real instrument, this could:
	• Trigger preventative maintenance
	• Prevent reporting bad results
	• Reduce false positives/negatives
	• Improve uptime

Avg Hit Score (MLP Model via ONNX)
This is a hit prioritization neural network running on:
	• CPU
	• Or iGPU (DirectML)
It simulates:
	AI scoring compounds or samples for likelihood of being a "true hit"
In real HTS pharma screening, this helps:
	• Reduce follow-up workload
	• Prioritize high-value samples
	• Improve downstream validation efficiency

Assay Value Trend Chart
Top chart shows:
	• Streaming assay values over time
	• Operational stability view
	• Drift detection visualization
In real lab terms:
This is what lab engineers watch to ensure assay consistency.

Hit Score vs Assay Value Scatter Plot
Bottom chart shows:
	• X-axis = AI hit confidence
	• Y-axis = Assay measurement
	• Red dots = anomalies
This visually demonstrates:
	AI can simultaneously score biological relevance AND detect operational risk.
This is powerful because traditional HTS systems treat these separately.

Speedup (CPU vs iGPU)
We are benchmarking:
	• CPU inference time
	• Intel Core Ultra iGPU (DirectML) inference time
This demonstrates:
	Edge AI acceleration without adding discrete GPUs.
For companies building next-gen instruments, this means:
	• No extra hardware cost
	• No cloud dependency
	• Real-time AI at the instrument
	• Lower latency
	• Lower bandwidth
	• Better cybersecurity

**Detailed Flow of the Reference Application**

1. AutoQC models (autoqc_train & autoqc_infer)
	• Purpose: Detect operational anomalies in the screening instrument. Think of this as real-time QC monitoring.
	• Flow:
		○ autoqc_train.py → trains the model on historical QC/instrument data.
		○ autoqc_infer.py → takes new incoming sample data (assay values, instrument readings, etc.) and predicts anomalies (anomaly_flag).
	• Dashboard role:
		○ Lower-left quadrant: Real-Time Anomaly Tracking.
		○ Shows red dots for flagged anomalies.
		○ Contributes to the “Anomalies” metric in the KPI bar.
	• Hardware: Runs on CPU, because AutoQC is lightweight.
	• Latency: Can measure CPU inference time per batch and display in workload table.

2. Drift models (drift_train & drift_infer)
	• Purpose: Monitor long-term drift in the assay values or instrument behavior.
	• Flow:
		○ drift_train.py → deep MLP trained to detect trends or deviations over time.
		○ drift_infer.py → computes drift scores on incoming batches.
	• Dashboard role:
		○ Could feed Operational Stability chart (top-left) showing flat vs sloped vs noisy trends.
	• Hardware: Runs on CPU, lightweight.
	• Latency: Usually very fast, can include as a baseline CPU workload.

3. Hit scoring models (hit_train & hit_infer)
	• Purpose: Rank samples by biological priority. This is the main AI ranking.
	• Flow:
		○ hit_train.py → trains deep MLP on assay features to predict hit scores.
		○ hit_infer.py → generates hit scores for each batch of samples.
	• Dashboard role:
		○ Top-right quadrant: Biological Prioritization chart.
		○ Top 1% hits highlighted, top candidates auto-labeled.
	• Hardware:
		○ Runs on iGPU (DML) to showcase accelerated inference.
		○ CPU can also run it but iGPU is preferred for demo purposes.
	• Latency: Measures iGPU throughput for Samples/hr metric; can be shown in workload table.

4. UI (ui.py or ui_v3_ADLM.py)
	• Purpose: Dashboard that ties everything together in real-time.
	• Flow:
		1. Generates new sample batch (batch_size rows).
		2. Sends batch to:
			§ AutoQC → anomaly predictions → lower-left chart & KPI.
			§ Hit scoring → hit scores → top-right chart & KPI.
			§ Drift → operational stability → top-left chart.
		3. Updates KPI bar (total samples, anomalies, top hits, Samples/hr).
		4. Updates workload table with CPU/iGPU tasks and optionally latency.
		5. Updates bottom-right chart (CPU/iGPU utilization).
	• Hardware visualization: Shows CPU utilization, iGPU utilization, and workload mapping.

How they all connect
Model / Script	Input	Output	Hardware	UI Component
autoqc_train.py	Historical QC data	Trained AutoQC model	CPU	N/A
autoqc_infer.py	New sample batch	anomaly_flag (-1 or 1)	CPU	Lower-left chart, KPI
drift_train.py	Historical assay/instrument data	Trained drift model	CPU	N/A
drift_infer.py	New sample batch	Drift score per sample	CPU	Top-left chart
hit_train.py	Sample features	Trained hit MLP model	CPU/iGPU	N/A
hit_infer.py	New sample batch	Hit scores per sample	iGPU	Top-right chart, KPI
ui.py	X_new batch	Calls AutoQC, Drift, Hit models	CPU/iGPU	All charts, KPI, workload table


