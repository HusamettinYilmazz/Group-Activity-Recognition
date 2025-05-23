import torch 
import torch.nn as nn
import torchvision.models as models


class SequentialGroupActivityPooledPersons(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.resnet50 = models.resnet50(weights="DEFAULT")
        self.lstm = nn.LSTM(self.resnet50.fc.in_features, hidden_size= 1024, batch_first= True)
        self.fc = nn.Sequential(
            nn.Linear(self.lstm.hidden_size, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, out_features)
        )

        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.max_pool = nn.AdaptiveMaxPool2d((1, 2048))  ## from (12, 2048) to (1, 2048)

    
    def forward(self, x):   ## x.shape = [batch_size, seq_length, num_players, 3, 244, 244] 
        batch_size, seq_length, num_players = x.shape[0], x.shape[1], x.shape[2]     ## batch_size = 2, seq_length = 9, num_players = 12
        
        x = x.view(-1, x.shape[3], x.shape[4], x.shape[5])  ## [(Batch_size X 9), c, w, h]
        resnet_out = self.resnet50(x).view(batch_size, seq_length, num_players, -1)    ## [(Batch_size X 9 X 12), 2048]  ---->  [Batch_size, seq_length=9, num_players=12, 2048]
        
        pooled_resnet_out = self.max_pool(resnet_out).view(batch_size, seq_length, -1)   ##[Batch_size, seq_length, 2048]

        lstm_out, (h, c) = self.lstm(pooled_resnet_out)  ## [Batch_size, time_stamps=seq_length,  1024]
        lstm_out = lstm_out[:, -1, :].view(batch_size, -1)   ##[Batch_size, 1024]

        preds = self.fc(lstm_out)
        return preds



if __name__ == "__main__":
    gar = SequentialGroupActivityPooledPersons(out_features= 8)
