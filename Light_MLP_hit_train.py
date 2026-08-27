import torch
import torch.nn as nn
import pandas as pd
import os

os.makedirs("models", exist_ok=True)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4,16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

data = pd.read_csv("hts_data.csv")
data["true_hit"] = (data["assay_value"] > 120).astype(int)

X = torch.tensor(
    data[["assay_value","qc_metric","instrument_temp","patient_delta"]].values,
    dtype=torch.float32
)
y = torch.tensor(data["true_hit"].values, dtype=torch.float32).view(-1,1)

model = MLP()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "models/hit_mlp.pt")
print("Hit prioritization model trained.")