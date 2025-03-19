import torch 
import torch.nn as nn
import torchvision.models as models


class GroupActivity(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.resnet50 = models.resnet50(weights="DEFAULT")
        self.resnet50.fc = nn.Linear(in_features= self.resnet50.fc.in_features, out_features=out_features)


    def forward(self, x):
        return self.resnet50(x)


if __name__ == "__main__":
    gar = GroupActivity(out_features= 8)
