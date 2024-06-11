import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_dim=1, embed_dim=24, hidden_dim=24, end_dim=128, num_layer=1, dropout=0.2, horizon=24):
        super(LSTM, self).__init__()
        self.start_conv = nn.Conv2d(in_channels=input_dim, 
                                    out_channels=embed_dim, 
                                    kernel_size=(1,1))

        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layer, batch_first=True, dropout=dropout)
        
        self.end_linear1 = nn.Linear(hidden_dim, end_dim)
        self.end_linear2 = nn.Linear(end_dim, horizon)


    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        x = history_data.transpose(1, 3)
        b, c, n, l = x.shape
        x_tmp = x.transpose(1,2).reshape(b*n, c, 1, l)
        x_in = self.start_conv(x_tmp).squeeze().transpose(1, 2)

        out, _ = self.lstm(x_in)

        x = out[:, -1, :] + torch.squeeze(x_tmp)

        x = F.relu(self.end_linear1(x))
        x = self.end_linear2(x)
        x = x.reshape(b, n, l, 1).transpose(1, 2)
        return x
