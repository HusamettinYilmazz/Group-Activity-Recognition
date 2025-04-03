import torch 
import torch.nn as nn
import torchvision.models as models


class SequentialGroupActivity(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.resnet50 = models.resnet50(weights="DEFAULT")
        self.lstm = nn.LSTM(self.resnet50.fc.in_features, hidden_size= 2048 , batch_first= True)
        self.fc = nn.Sequential(
            nn.Linear(self.lstm.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_features)
        )
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])


    def forward(self, x):
        batch_size, seq_length = x.shape[0], x.shape[1]
        
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])  ## [(Batch_size X 9), 3, w, h]
        resnet_out = self.resnet50(x).view(batch_size, seq_length, -1)    ## [(Batch_size X 9), 2048]  ---->  [Batch_size, 9, 2048]
        # resnet_out = resnet_out.view(batch_size, seq_length, -1)  ## [Batch_size, 9, 2048]
        
        lstm_out, (h, c) = self.lstm(resnet_out)  ## [Batch_size, 2048]
        
        preds = self.fc(lstm_out[:, -1, :])   ## [Batch_size, 1]

        return preds


if __name__ == "__main__":
    gar = SequentialGroupActivity(out_features= 8)
