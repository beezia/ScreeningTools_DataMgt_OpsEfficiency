import numpy as np
import time
import onnxruntime as ort

print("Available Providers:", ort.get_available_providers())

gpu_session = ort.InferenceSession(
    "models/hit_transformer.onnx",
    providers=["DmlExecutionProvider","CPUExecutionProvider"]
)

cpu_session = ort.InferenceSession(
    "models/hit_transformer.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = gpu_session.get_inputs()[0].name

batch_sizes = [256, 512, 1024, 2048, 4096]

print("\nBenchmarking CPU vs iGPU\n")

for batch in batch_sizes:

    X = np.random.normal(0,1,(batch,64)).astype(np.float32)

    # CPU timing
    start = time.time()
    for _ in range(10):
        cpu_session.run(None, {input_name: X})
    cpu_time = (time.time() - start)/10

    # GPU timing
    start = time.time()
    for _ in range(10):
        gpu_session.run(None, {input_name: X})
    gpu_time = (time.time() - start)/10

    speedup = cpu_time/gpu_time if gpu_time > 0 else 0

    print(f"Batch {batch}")
    print(f"CPU Latency: {cpu_time:.4f} sec")
    print(f"iGPU Latency: {gpu_time:.4f} sec")
    print(f"Acceleration: {speedup:.2f}x\n")