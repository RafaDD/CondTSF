import random
import time
import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import TensorDataset, DataLoader


def log_verbose(args, msg):
    print(msg)
    with open(os.path.join(args.save_dir, "log.txt"), "a") as f:
        f.write(msg + "\n")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)
    return None

def calc_metric(pred, y, stage = "train"):
    if(stage == "train"):
        MSE = torch.mean((pred - y)**2)
        MAE = torch.mean(torch.abs(pred - y))
    else:
        B, N, L = pred.shape
        pred = pred.reshape(-1, L)
        y = y.reshape(-1, L)
        MSE = torch.mean((pred - y)**2, dim = 0)
        MAE = torch.mean(torch.abs(pred - y), dim = 0)
    return MSE.detach().cpu().numpy(), MAE.detach().cpu().numpy()

def generate_dataset(X, num_timesteps_input, num_timesteps_output, inter_step, syn=False):
    X = X[:, :, :]

    if(not syn):
        X = X.numpy()
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i in \
        range(0, X.shape[2] - (num_timesteps_input + num_timesteps_output) + 1, inter_step)]

    features, target = [], []
    for i, j in indices:
        if not syn:
            features.append(X[:, :, i: i + num_timesteps_input].transpose((0, 2, 1)))
        else:
            features.append(X[:,:,i:i+num_timesteps_input].permute(0,2,1))
        target.append(X[:, 0, i + num_timesteps_input: j])
    if not syn:
        x = torch.from_numpy(np.array(features)).float()
        y = torch.from_numpy(np.array(target)).float()
    else:
        x = torch.stack(features, dim=0)
        y = torch.stack(target, dim=0)
    return x, y


def epoch(mode, data, net, optimizer, criterion, args):
    loss_avg = 0
    MSE_total, MAE_total = 0, 0
    num_b, num_sample = 0, 0
    net = net.to(args.device)
    data_loader, model_name = data['data_loader'], data['model']
    if('train' in mode):
        net.train()
    else:
        net.eval()

    for i_batch, (X, y) in enumerate(data_loader):
        y_pred = None

        if ('train' in mode) and ('ftd' in mode):
            def loss_fn(pred, target):
                return criterion(pred, target)
            
            optimizer.set_closure(loss_fn, X.permute(0, 2, 1, 3), y)
            y_pred, loss = optimizer.step()
                
        if model_name == 'DLinear':
            y_pred = net(X.permute(0, 2, 1, 3))
        elif model_name == 'MLP':
            y_pred = net(X.permute(0, 2, 1, 3))
            y_pred = y_pred[..., 0].permute(0, 2, 1)
        elif model_name == 'LSTM':
            y_pred = net(X.permute(0, 2, 1, 3))
            y_pred = y_pred[..., 0].permute(0, 2, 1)
        elif model_name == 'CNN':
            y_pred = net(X)
        else:
            raise NotImplementedError
        
        loss = criterion(y_pred, y)
        n_b = y.shape[0]
        loss_avg += loss.item() * n_b
        
        n_sample = torch.prod(torch.tensor(y.shape)).detach().cpu().numpy()
        MSE, MAE = calc_metric(y_pred, y, stage = 'train')
        MSE_total += MSE * n_sample
        MAE_total += MAE * n_sample
        
        num_b += n_b
        num_sample += n_sample

        if ('train' in mode) and ('ftd' not in mode):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    MSE_total /= num_sample
    MAE_total /= num_sample
    loss_avg /= num_b
        
    return loss_avg, MSE_total, MAE_total

def evaluate_synset(it_eval, net, GCond, real_data_dict, args, input_len, output_len, use_data='val', return_loss=False):
    net = net.to(args.device)
    GCond.eval()
    syn_data = GCond.feat_syn.detach().to('cpu')
    lr = float(args.lr_teacher)
    Epoch = int(args.epoch_eval_train)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=args.net_mom)
    
    criterion = nn.MSELoss(reduction='mean').to(args.device)
    
    inter_step = 4
    syn_X, syn_Y = generate_dataset(syn_data, input_len, output_len, inter_step, syn=True)
    syn_X, syn_Y = syn_X.to(args.device), syn_Y.to(args.device)
    syn_dataset = TensorDataset(syn_X, syn_Y)
    syn_loader = DataLoader(syn_dataset,batch_size=args.batch_train,shuffle=True)
    
    test_loader = real_data_dict[use_data]
    
    time0 = time.time()
    MSE_list, MAE_list = [],[]
    
    for ep in range(Epoch+1):
        train_loss, train_MSE, train_MAE = epoch('train_evalsyn', data={'data_loader': syn_loader, 'model': args.model, 'GCond':GCond}, net=net, optimizer=optimizer, criterion=criterion, args=args)
        if(ep % 50 == 0 or ep == Epoch):
            log_verbose(args,"Evaluation SynSet id : {}, Time : {:.5f}\t Epoch: {}/{}\tTrain MSE: {:.4f}\tMAE: {:.4f}".format(it_eval, time.time()-time0, ep,Epoch, train_MSE, train_MAE))
        
        MSE_list.append(train_MSE)
        MAE_list.append(train_MAE)
        time0 = time.time()
            
        if(ep == Epoch):
            test_loss, test_MSE, test_MAE = epoch('test_evalsyn', data={'data_loader': test_loader, 'model': args.model,'GCond': GCond}, net = net, optimizer=None, criterion=criterion, args=args)
            res_str = "Evaluation SynSet id : {}, Epoch: {}\tTest MSE: {:.4f}\tMAE: {:.4f}".format(it_eval, ep, test_MSE, test_MAE)
            log_verbose(args, res_str)
        
    if return_loss:
        return net, MSE_list, MAE_list, test_MSE, test_MAE, train_loss, test_loss
    else:
        return net, MSE_list, MAE_list, test_MSE, test_MAE