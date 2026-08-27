import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

os.makedirs("models", exist_ok=True)

np.random.seed(42)

# ----------------------------
# Simulated HTS Dataset
# ----------------------------

samples = 60000

X = np.random.normal(0,1,(samples,4))

signal = (
    0.8*X[:,0]
    -0.6*X[:,1]
    +0.5*X[:,2]
    +0.2*np.sin(X[:,3])
)

activity = (signal + np.random.normal(0,0.5,samples)) > 0.7

# novelty = rare chemical patterns

novelty = np.abs(X[:,0]*X[:,2]) > 1.5

y_activity = activity.astype(float)
y_novelty = novelty.astype(float)

X = torch.tensor(X,dtype=torch.float32)
y_activity = torch.tensor(y_activity,dtype=torch.float32).unsqueeze(1)
y_novelty = torch.tensor(y_novelty,dtype=torch.float32).unsqueeze(1)

# ----------------------------
# Dual Head Deep MLP
# ----------------------------

class HitModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.shared = nn.Sequential(

            nn.Linear(4,512),
            nn.ReLU(),

            nn.Linear(512,512),
            nn.ReLU(),

            nn.Linear(512,256),
            nn.ReLU()
        )

        # Activity head
        self.activity_head = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

        # Novelty head
        self.novelty_head = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

    def forward(self,x):

        shared = self.shared(x)

        activity = self.activity_head(shared)
        novelty = self.novelty_head(shared)

        return activity, novelty


model = HitModel()

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# Training Loop
# ----------------------------

for epoch in range(25):

    optimizer.zero_grad()

    act_pred, nov_pred = model(X)

    loss1 = criterion(act_pred,y_activity)
    loss2 = criterion(nov_pred,y_novelty)

    loss = loss1 + loss2

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1} Loss {loss.item():.4f}")

torch.save(model.state_dict(),"models/hit_dual_mlp.pt")

print("Training complete")