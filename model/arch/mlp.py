import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, **model_args):
        super(MLP, self).__init__()
        self.seq_len = 24
        self.pred_len = 24
        self.Linear = nn.Sequential(nn.Linear(self.seq_len, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, self.pred_len))

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        assert history_data.shape[-1] == 1
        x = history_data[..., 0]
        x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        return x.unsqueeze(-1)