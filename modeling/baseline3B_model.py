import torch 
import torch.nn as nn
import torchvision.models as models
from baseline1_model import GroupActivity


MODEL_ROOT = '/teamspace/studios/this_studio/Group-Activity-Recognition/modeling/outputs/Baseline 3/V1.0/epoch8_model.pth'

class GroupActivity3B(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.back_bone = self.load_pretrained_model()
        self.fc = nn.Sequential(
            nn.Linear(in_features= self.back_bone.resnet50.fc.in_features, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=out_features)
            )
        
        self.back_bone = nn.Sequential(*list(self.back_bone.resnet50.children())[:-1])
        for param in self.back_bone.parameters():
            param.requires_grad = False
        self.max_pool = nn.AdaptiveMaxPool2d((1, 2048))  ## from (12, 2048) to (1, 2048)
        

    def forward(self, x):
        batch_size, seq_length = x.shape[0], x.shape[1]
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])  ## [(batch_size * seq_length), c, h, w]
        feat_vector = self.back_bone(x)         ## [(batch_size * seq_length), 2048, 1, 1]
        feat_vector = feat_vector.view(batch_size, seq_length, -1)  ## [batch_size, seq_length, 2048]
        pooled_features = self.max_pool(feat_vector)    ## [batch_size, 1, 2048]
        pooled_features = pooled_features.view(batch_size, -1)  ## [batch_size, 2048]
        out = self.fc(pooled_features)      ## [batch_size, num_classes]

        return out
            
    def load_pretrained_model(self):
        check_point = torch.load(MODEL_ROOT, map_location= torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model = GroupActivity(out_features=9)
        model.load_state_dict(check_point['model'])
        
        return model
    

if __name__ == "__main__":
    gar = GroupActivity3B(out_features= 8)

