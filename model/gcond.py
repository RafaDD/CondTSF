import torch
import torch.nn as nn
import os
from utils import *

class GCOND(nn.Module):
    def __init__(self, mode, true_series, args, device='cuda'):
        super(GCOND, self).__init__()
        self.args = args
        self.device = device
        
        if mode != 'oneshot':
            self.Dist_N, self.Dist_L = args.N, int(args.L * args.TS_CompRate)
        else:
            self.Dist_N, self.Dist_L = args.N, 50
        
        self.true_series = true_series
        self.feat_syn = nn.Parameter(torch.FloatTensor(self.Dist_N, 1, self.Dist_L).to(device))

        self.reset_parameters()
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_feat.zero_grad()
        
        log_verbose(args,'feat_sym: ({})'.format(self.feat_syn.shape))
        
    def reset_parameters(self):
        # use true data to initialize the synthetic data
        true_series = torch.tensor(self.true_series, dtype=torch.float).to(self.args.device)
        N_true, _, L_true = true_series.shape
        select_node = torch.randperm(N_true)[:self.Dist_N]
        
        # Original random select
        select_start = torch.randint(0, L_true - self.Dist_L, (1,))
        
        # Fixed start selection
        # select_start = int((L_true - self.Dist_L) / 2)
        select_series = true_series[select_node,:,select_start:select_start + self.Dist_L]
        self.feat_syn.data.copy_(select_series)
            

    def save_syn(self, save_dir, name):
        torch.save(self.feat_syn.detach().cpu().numpy(), os.path.join(save_dir,'feat_{}'.format(name)))

        
    def load_best_syn(self, save_dir):
        syn_feat = torch.from_numpy(torch.load(os.path.join(save_dir,'feat_best')))
        self.feat_syn = nn.Parameter(syn_feat.to(self.args.device))

