import torch
import torch.nn as nn


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

        x = torch.add(x,attn_out)

        x = x.transpose(1,2)
        x = self.norm1(x)
        x = x.transpose(1,2)

        ff_out = self.ff(x)

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

model.load_state_dict(torch.load("models/hit_transformer.pt"))

model.eval()

dummy = torch.randn(1024,64)

torch.onnx.export(
    model,
    dummy,
    "models/hit_transformer.onnx",
    input_names=["model_input"],
    output_names=["output"],
    opset_version=17,
    do_constant_folding=False
)

print("ONNX export complete")