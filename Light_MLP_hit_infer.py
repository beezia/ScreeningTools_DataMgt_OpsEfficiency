import torch
import onnxruntime as ort
import numpy as np
import pandas as pd

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4,16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16,1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))

# Load model
model = MLP()
model.load_state_dict(torch.load("models/hit_mlp.pt"))
model.eval()

# Export to ONNX (explicit naming)
dummy = torch.randn(1,4)

torch.onnx.export(
    model,
    dummy,
    "models/hit_mlp.onnx",
    input_names=["model_input"],
    output_names=["model_output"],
    opset_version=17,
    dynamic_axes={
        "model_input": {0: "batch_size"},
        "model_output": {0: "batch_size"}
    }
)

# Load data
data = pd.read_csv("hts_data.csv")
X = data[["assay_value","qc_metric","instrument_temp","patient_delta"]].values.astype(np.float32)

# Create session
session = ort.InferenceSession(
    "models/hit_mlp.onnx",
    providers=["DmlExecutionProvider", "CPUExecutionProvider"]
)

# 🔎 Get actual input name dynamically
input_name = session.get_inputs()[0].name

# Run inference
outputs = session.run(None, {input_name: X})

print("Inference complete. Top scores:")
print(outputs[0][:10])