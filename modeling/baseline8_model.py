import torch 
import torch.nn as nn
import torchvision.models as models


class SequentialGroupActivityPooledPersons(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.resnet50 = models.resnet50(weights="DEFAULT")
        
        self.lstm1 = nn.LSTM(self.resnet50.fc.in_features, hidden_size= 512, batch_first= True)
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1024))  ## from (6, 1024) to (1, 1024)
        self.layer_norm = nn.LayerNorm(2048)
        
        self.lstm2 = nn.LSTM(self.max_pool.output_size[1] * 2, hidden_size= 512, batch_first= True)
        
        self.fc = nn.Sequential(
            nn.Linear(self.lstm2.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, out_features)
        )

    

    def forward(self, x):   ## x.shape = [batch_size, seq_length, num_players, 3, 244, 244] 
        batch_size, seq_length, num_players = x.shape[0], x.shape[1], x.shape[2]     ## batch_size = 16, seq_length = 9, num_players = 12
        
        x = x.view(-1, x.shape[3], x.shape[4], x.shape[5])  ## [(Batch_size X 9), 3, w, h]
        resnet_out = self.resnet50(x).view(batch_size, seq_length, num_players, -1)    ## [(Batch_size X 9 X 12), 2048]  ---->  [Batch_size, seq_length=9, num_players=12, 2048]
        resnet_out = resnet_out.permute(0, 2, 1, 3)  ## [Batch_size, players, frames, 2048]
        resnet_out = resnet_out.reshape(batch_size * num_players, seq_length, -1) ## [Batch_size * players, frames, 2048]
        
        
        resnet_out = self.layer_norm(resnet_out)
        lstm1_out, (h1, c1) = self.lstm1(resnet_out)    ## lstm1_out = [batch_size * num_players=24, frames=9, 1024]

        lstm1_out = torch.cat((lstm1_out, resnet_out), dim=2)   ## [batch_size * num_players = 24, frames=9, 3072]
        
        lstm1_out = lstm1_out.view(batch_size, num_players, seq_length,  -1)   ##[Batch_size, players, hidden_states = frames =9,  3072]
        lstm1_out = lstm1_out.permute(0, 2, 1, 3).reshape(batch_size * seq_length, num_players, -1)   ## [Batch_size * seq_length =9, players,  3072]
        
        l_team = lstm1_out[:, :6, :]
        r_team = lstm1_out[:, 6:, :]

        pooled_l_team = self.max_pool(l_team)
        pooled_r_team = self.max_pool(r_team)

        
        pooled_all_players = torch.cat((pooled_l_team, pooled_r_team), dim=2).view(batch_size, seq_length, -1)  ##[Batch_size, hidden_states = frames =9, 1024]
        pooled_all_players = self.layer_norm(pooled_all_players)        
        
        lstm2_out, (h2, c2) = self.lstm2(pooled_all_players)
        
        preds = self.fc(lstm2_out[:, -1, :].view(batch_size, -1))
        
        return preds


if __name__ == "__main__":
    gar = SequentialGroupActivityPooledPersons(out_features= 8)
