import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

os.makedirs("models", exist_ok=True)

np.random.seed(42)

X = np.random.normal(0,1,(15000,64))
y = (np.sum(X[:,:8],axis=1) > 2).astype(int)

X_tensor = torch.tensor(X,dtype=torch.float32)
y_tensor = torch.tensor(y,dtype=torch.long)


class SafeTransformerBlock(nn.Module):

    def __init__(self,embed_dim=64,num_heads=4):
        super().__init__()

        self.attn = nn.MultiheadAttention(embed_dim,num_heads,batch_first=True)

        self.norm1 = nn.GroupNorm(1,embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim,128),
            nn.ReLU(),
            nn.Linear(128,embed_dim)
        )

        self.norm2 = nn.GroupNorm(1,embed_dim)

    def forward(self,x):

        attn_out,_ = self.attn(x,x,x)

        attn_out = attn_out.contiguous()
        x = x.contiguous()

        x = torch.add(x,attn_out)

        x = x.transpose(1,2)
        x = self.norm1(x)
        x = x.transpose(1,2)

        ff_out = self.ff(x)

        ff_out = ff_out.contiguous()
        x = x.contiguous()

        x = torch.add(x,ff_out)

        x = x.transpose(1,2)
        x = self.norm2(x)
        x = x.transpose(1,2)

        return x


class TabTransformer(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Linear(1,64)

        self.block1 = SafeTransformerBlock()
        self.block2 = SafeTransformerBlock()

        self.fc = nn.Sequential(
            nn.Linear(64*64,256),
            nn.ReLU(),
            nn.Linear(256,2)
        )

    def forward(self,x):

        x = x.unsqueeze(-1)

        x = self.embedding(x)

        x = self.block1(x)
        x = self.block2(x)

        x = x.flatten(start_dim=1)

        return self.fc(x)


model = TabTransformer()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)


for epoch in range(10):

    optimizer.zero_grad()

    outputs = model(X_tensor)

    loss = criterion(outputs,y_tensor)

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1} Loss: {loss.item()}")

torch.save(model.state_dict(),"models/hit_transformer.pt")

print("Hit transformer training complete")