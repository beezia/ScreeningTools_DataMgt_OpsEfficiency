import onnxruntime as ort
import numpy as np
import time

# Larger batch to highlight iGPU benefit
X = np.random.rand(20000,4).astype(np.float32)

# ---------------- CPU SESSION ----------------
cpu_session = ort.InferenceSession(
    "models/hit_mlp.onnx",
    providers=["CPUExecutionProvider"]
)

input_name_cpu = cpu_session.get_inputs()[0].name

start = time.time()
cpu_session.run(None, {input_name_cpu: X})
cpu_time = time.time() - start

# ---------------- iGPU SESSION ----------------
gpu_session = ort.InferenceSession(
    "models/hit_mlp.onnx",
    providers=["DmlExecutionProvider"]
)

input_name_gpu = gpu_session.get_inputs()[0].name

start = time.time()
gpu_session.run(None, {input_name_gpu: X})
gpu_time = time.time() - start

print(f"\nBatch size: {X.shape[0]}")
print(f"CPU time:  {cpu_time:.6f} sec")
print(f"iGPU time: {gpu_time:.6f} sec")
print(f"Speedup:   {cpu_time/gpu_time:.2f}x")