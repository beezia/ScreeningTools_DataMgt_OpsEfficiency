import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

os.makedirs("models", exist_ok=True)

np.random.seed(42)

plates = 800
wells_per_plate = 96
samples = plates * wells_per_plate

features = []

activity_labels = []
novelty_labels = []

for plate in range(plates):

    drift = np.random.normal(0,0.2)

    for well in range(wells_per_plate):

        row = well // 12
        col = well % 12

        edge = (row in [0,7] or col in [0,11])

        x = np.random.normal(0,1,4)

        # edge effect
        if edge:
            x += np.random.normal(0.3,0.1,4)

        # plate drift
        x += drift

        signal = (
            0.9*x[0]
            -0.7*x[1]
            +0.4*x[2]
            +0.2*np.sin(x[3])
        )

        active = signal + np.random.normal(0,0.6) > 1.0

        novelty = abs(x[0]*x[2]) > 1.8

        features.append(x)

        activity_labels.append(active)
        novelty_labels.append(novelty)

X = torch.tensor(np.array(features),dtype=torch.float32)

y_activity = torch.tensor(activity_labels,dtype=torch.float32).unsqueeze(1)
y_novelty = torch.tensor(novelty_labels,dtype=torch.float32).unsqueeze(1)

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

        self.activity_head = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

        self.novelty_head = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

    def forward(self,x):

        shared = self.shared(x)

        activity = self.activity_head(shared)
        novelty = self.novelty_head(shared)

        return activity,novelty


model = HitModel()

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(25):

    optimizer.zero_grad()

    act_pred,nov_pred = model(X)

    loss = criterion(act_pred,y_activity) + criterion(nov_pred,y_novelty)

    loss.backward()

    optimizer.step()

    print("Epoch",epoch+1,"loss",loss.item())

torch.save(model.state_dict(),"models/hit_dual_mlp.pt")

print("Training complete")