import torch
import torch.nn as nn
import joblib
import numpy as np
import os

class DeepDriftMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.model(x)

model = DeepDriftMLP()
model.load_state_dict(torch.load("models/drift_deep_mlp.pt"))
model.eval()

dummy = torch.randn(1, 32)

torch.onnx.export(
    model,
    dummy,
    "models/drift_deep_mlp.onnx",
    input_names=["model_input"],
    output_names=["output"],
    dynamic_axes={"model_input": {0: "batch_size"}}
)

print("Drift model exported to ONNX")