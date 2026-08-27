import torch
import torch.nn as nn

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

        return activity, novelty


model = HitModel()

model.load_state_dict(torch.load("models/hit_dual_mlp.pt"))

model.eval()

dummy = torch.randn(256,4)

torch.onnx.export(
    model,
    dummy,
    "models/hit_mlp.onnx",
    input_names=["input"],
    output_names=["activity","novelty"],
    dynamic_axes={"input":{0:"batch"}},
    opset_version=17
)

print("ONNX export complete")