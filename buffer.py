import os
import argparse
import torch
import torch.nn as nn
from utils import *
from torch.utils.data import TensorDataset, DataLoader
from model import STmodels
import numpy as np
import warnings
import time
warnings.filterwarnings("ignore", category=DeprecationWarning)
set_seed(7)

def main(args):
    if args.FTD == 0:
        args.save_dir = f'./buffers/{args.dataset}/{args.model}/'
    else:
        args.save_dir = f'./buffers/{args.dataset}/{args.model}_FTD/'
        
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    log_verbose(args, "Args : {}\n----------------".format(args.__dict__))
    
    TS = torch.from_numpy(np.load(f'./data/{args.dataset}/{args.dataset}.npy').T).float()
    input_len = 24
    output_len = 24  
    
    TS = (TS - torch.unsqueeze(torch.mean(TS, -1), -1)) / torch.unsqueeze(torch.std(TS, -1), -1)
    TS = torch.unsqueeze(TS, 1)
        
    N, L = TS.shape[0], TS.shape[2]
    
    devide = {'train':[0,int(L*0.7)], 'val':[int(L*0.7),int(L*0.7)], 'test':[int(L*0.7), L]}
    data_devided = {}
    for k,v in devide.items():
        data_devided[k] = {}
        data_devided[k]['X'], data_devided[k]['Y'] = generate_dataset(TS[:,:,v[0]:v[1]], input_len, output_len, 4)
        data_devided[k]['X'], data_devided[k]['Y'] = data_devided[k]['X'].to(args.device), data_devided[k]['Y'].to(args.device)
    
    data_sets, data_loaders = {}, {}
    for k,v in data_devided.items():
        data_sets[k] = TensorDataset(v['X'].detach(),v['Y'].detach())
        print('{} : X is {}, Y is {}'.format(k, v['X'].shape, v['Y'].shape))
        if(k=='val'):
            continue
        data_loaders[k] = DataLoader(data_sets[k], batch_size=args.batch_train, shuffle=True)
    
    
    save_dir = os.path.join(args.buffer_path, args.dataset)
    if args.FTD == 0:
        mode_prefix = ''
        save_dir = os.path.join(save_dir, args.model)
    else:
        mode_prefix = 'ftd_'
        save_dir = os.path.join(save_dir, args.model + '_FTD')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    criterion = nn.MSELoss(reduction='mean').to(args.device)
    trajectories = []
    
    for it in range(0, args.num_experts):
        print("======================")
        print("Trajectory ID : {}".format(it))
        
        teacher_net = STmodels.get_network(args.model).to(args.device)
        teacher_net.train()
        lr = args.lr_teacher
        
        if args.FTD == 0:
            teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
        else:
            from gsam import GSAM, ProportionScheduler
            base_optimizer = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
            scheduler = torch.optim.lr_scheduler.StepLR(base_optimizer, step_size=args.train_epochs, gamma=1)
            rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=lr, min_lr=lr,
                max_value=2.0, min_value=2.0)
            teacher_optim = GSAM(params=teacher_net.parameters(), base_optimizer=base_optimizer, 
                    model=teacher_net, gsam_alpha=0.4, rho_scheduler=rho_scheduler, adaptive=True)
        
        teacher_optim.zero_grad()
        timestamps = []
        timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])
        
        time0 = time.time()
        for e in range(args.train_epochs):
            _, train_MSE, train_MAE = epoch(mode_prefix+'train', data={'data_loader': data_loaders['train'], 'model': args.model}, net=teacher_net, optimizer=teacher_optim, criterion=criterion, args=args)
            _, test_MSE, test_MAE = epoch(mode_prefix+'test', data={'data_loader': data_loaders['test'], 'model': args.model}, net=teacher_net, optimizer=teacher_optim, criterion=criterion, args=args)
            
            log_verbose(args, "Epoch: {}\tTrain MSE: {:.4f}\tMAE: {:.4f}".format(e, train_MSE, train_MAE))
            log_verbose(args, "Epoch: {}\tTest MSE: {:.4f}\tMAE: {:.4f}".format(e, test_MSE, test_MAE))
            print("Cost : {:.3f}s".format(time.time() - time0))
            time0 = time.time()

            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])
            
        trajectories.append(timestamps)
        
        if len(trajectories) == args.save_interval:
            n = 0
            while(os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))):
                n += 1
            print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
            trajectories = []
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='ETTh1', help='dataset')
    parser.add_argument('--model', type=str, default='DLinear', help='model')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--batch_train', type=int, default=64, help='batch size for training networks')
    parser.add_argument('--num_experts', type=int, default=40, help='training iterations')
    parser.add_argument('--lr_teacher', type=float, default=5e-4, help='learning rate for updating network parameters')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--train_epochs', type=int, default=80)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--FTD', type=int, default=0)

    args = parser.parse_args()
    main(args)

