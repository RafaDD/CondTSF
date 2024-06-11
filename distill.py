import os
import argparse
import torch
import torch.nn as nn
from utils import *
from torch.utils.data import TensorDataset, DataLoader
from model import STmodels, gcond
import copy
import numpy as np
import warnings
from torchreparam import ReparamModule
import warnings
import datetime
from backbone import *
warnings.filterwarnings("ignore", category=DeprecationWarning)


condensing_rate = {
    'ETTh1': 4e-3, 'ETTh2': 4e-3, 'ETTm1': 2e-3, 'ETTm2': 2e-3,
    'ExchangeRate': 1e-2, 'Weather': 2e-3, 'Electricity': 3e-3, 'Traffic': 4e-3
}

def main(args):
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.curr_time = curr_time
    framework = args.framework
    if args.cond_gap > args.Iteration:
        save_dir = os.path.join(".", "logged_files_baseline", framework, args.dataset, args.curr_time)
    else:
        save_dir = os.path.join(".", "logged_files", framework, args.dataset, args.curr_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir
    
    args.curr_time = curr_time
    
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    log_verbose(args,"Args : {}\n----------------".format(args.__dict__))

    TS = torch.from_numpy(np.load(f'./data/{args.dataset}/{args.dataset}.npy').T).float()
    input_len = 24
    output_len = 24
    
    TS = (TS - torch.unsqueeze(torch.mean(TS, -1), -1)) / torch.unsqueeze(torch.std(TS, -1), -1)
    TS = torch.unsqueeze(TS, 1)
    N, L = TS.shape[0], TS.shape[2]
    args.N, args.L = N, L
    
    devide = {'train':[0, int(L*0.7)], 'val':[int(L*0.7), L], 'test':[int(L*0.7), L]}
    data_devided = {}
    for k,v in devide.items():
        data_devided[k] = {}
        data_devided[k]['X'], data_devided[k]['Y'] = generate_dataset(TS[:,:,v[0]:v[1]], input_len, output_len, 12)
        data_devided[k]['X'], data_devided[k]['Y'] = data_devided[k]['X'].to(args.device), data_devided[k]['Y'].to(args.device)
    
    data_sets, data_loaders = {}, {}
    for k, v in data_devided.items():
        data_sets[k] = TensorDataset(v['X'].detach(),v['Y'].detach())
        log_verbose(args,'{} : X is {}, Y is {}'.format(k, v['X'].shape, v['Y'].shape))
        data_loaders[k] = DataLoader(data_sets[k], batch_size = args.batch_train, shuffle = True)
    
    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    
    model_eval_pool = ['DLinear']
    
    log_verbose(args,"Eval_it_pool : {}\tmodel_eval_pool : {}".format(eval_it_pool, model_eval_pool))
    
    if args.condensing_ratio == 'standard':
        args.TS_CompRate = condensing_rate[args.dataset]
    elif args.condensing_ratio == '3-standard':
        args.TS_CompRate = condensing_rate[args.dataset] * 3
    
    GCond = gcond.GCOND(args.condensing_ratio, TS[:,:,devide['train'][0]:devide['train'][1]], args, device=args.device)
    criterion = nn.MSELoss().to(args.device)
    
    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.framework == 'FTD':
        expert_dir = os.path.join(expert_dir, args.model + '_FTD')
    else:
        expert_dir = os.path.join(expert_dir, args.model)
    log_verbose(args,"Expert Dir : {}".format(expert_dir))
    
    if args.framework == 'IDM':
        Q_model_pool = []
        Q_max_len = 8
    
    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        
        start_trajec = 0
        if args.framework == 'DATM':
            end_trajec = 2
        else:
            end_trajec = args.max_start_epoch
        
        log_verbose(args,"loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        random.shuffle(buffer)
    
    best_MSE = {m: 99999999.0 for m in model_eval_pool}
    best_MSE_std = {m: 99999999.0 for m in model_eval_pool}
    
    for it in range(0, args.Iteration + 1):
        save_this_it = False
        
        condition = it in eval_it_pool

        if condition:
            for model_eval in model_eval_pool:
                log_verbose(args,'-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                
                MSE_train, MAE_train, = [], []
                MSE_test, MAE_test, = [], []
                
                for it_eval in range(args.num_eval):
                    net_eval = STmodels.get_network(args.model).to(args.device)

                    eval_GCond = copy.deepcopy(GCond)
                    __, MSE_list, MAE_list, test_MSE, test_MAE = evaluate_synset(it_eval=it_eval, net=net_eval, GCond=eval_GCond, input_len=input_len,
                                                                                output_len=output_len, real_data_dict = data_loaders, args=args, use_data='val')
                    
                    MSE_train.append(MSE_list)
                    MAE_train.append(MAE_list)
                    MSE_test.append(test_MSE)
                    MAE_test.append(test_MAE)
                    
                MSE_train = np.array(MSE_train)
                MAE_train = np.array(MAE_train)
                MSE_test = np.array(MSE_test)
                MAE_test = np.array(MAE_test)
                
                log_verbose(args,"Eval MSE: {:.5f} +/- {:.5f}".format(MSE_test.mean(), MSE_test.std()))
                log_verbose(args,"Eval MAE: {:.5f} +/- {:.5f}".format(MAE_test.mean(), MAE_test.std()))
                
                if MSE_test.mean() < best_MSE[model_eval]:
                    best_MSE[model_eval] = MSE_test.mean()
                    best_MSE_std[model_eval] = MSE_test.std()
                    save_this_it = True

                    log_verbose(args,"This is the best model so far!")    
                    
        if save_this_it or it in eval_it_pool:
            with torch.no_grad():
                GCond.save_syn(save_dir, it)
                if save_this_it:
                    GCond.save_syn(save_dir, 'best')
                    log_verbose(args,"Synthetic Training Epoch : {}, Saved Best!".format(it))
            save_this_it = False
        
        student_net = STmodels.get_network(args.model).to(args.device)
        student_net = ReparamModule(student_net)
        student_net.train()

        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                
                log_verbose(args,"loading file {}".format(expert_files[file_idx]))
                del buffer
                buffer = torch.load(expert_files[file_idx])
                random.shuffle(buffer)
            expert_trajectory = buffer[expert_idx]
        
        if args.framework == 'DATM':
            if it % 40 == 0 and it != 0:
                start_trajec = min(start_trajec+1, args.max_start_epoch+8)
                end_trajec = min(end_trajec+1, args.max_start_epoch+12)

        condition = ((it + 1) % args.cond_gap) != 0
        
        if condition:
            GCond.feat_syn.requires_grad = True
        else:
            GCond.feat_syn.requires_grad = False
            
        start_epoch = np.random.randint(start_trajec, end_trajec)
        
        for multi_step in range(args.multistep):
            syn_data = GCond.feat_syn
            inter_step = 4
            
            syn_X_total, syn_Y_total = generate_dataset(syn_data, input_len, input_len, inter_step,syn=True)
            syn_X_total, syn_Y_total = syn_X_total.to(args.device), syn_Y_total.clone().to(args.device).detach()
            
            starting_params = expert_trajectory[start_epoch].copy()
            target_params = expert_trajectory[start_epoch + args.expert_epochs].copy()
            target_params = torch.cat([p.data.reshape(-1).to(args.device) for p in target_params], 0)
            
            student_params = torch.cat([p.data.reshape(-1).to(args.device) for p in starting_params], 0).requires_grad_(True)
            starting_params = torch.cat([p.data.reshape(-1).to(args.device) for p in starting_params], 0)
            param_dist = torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

            for step in range(args.syn_steps):
                b = args.batch_syn
                perm = torch.randperm(syn_X_total.shape[0])[:b]
                syn_X = syn_X_total[perm,:,:]
                syn_Y = syn_Y_total[perm,:,:]
                
                syn_X = syn_X.permute(0, 2, 1, 3)
                target_Y = student_net(syn_X, flat_param=target_params)
                y_pred = student_net(syn_X, flat_param=student_params)
                    
                syn_Y = syn_Y * args.alpha + target_Y * (1 - args.alpha)

                mse_loss = criterion(y_pred, syn_Y)
                if step == args.syn_steps - 1:
                    params_tmp = expert_trajectory[-1].copy()
                    params_tmp = torch.cat([p.data.reshape(-1).to(args.device) for p in params_tmp], 0)
                    target_Y = student_net(syn_X, flat_param=params_tmp)
                    y_loss = torch.nn.functional.mse_loss(target_Y, syn_Y_total[perm,:,:], reduction="mean")
                
                grad = torch.autograd.grad(mse_loss, student_params, create_graph=True, allow_unused=True)[0]
                
                student_params = student_params - args.lr_init * grad
            
            param_loss = torch.nn.functional.mse_loss(student_params, target_params, reduction="sum")
            param_loss /= param_dist

            log_verbose(args,"Synthetic Training Epoch : {}, step : {}, Param loss : {:.4f}, Label loss : {:.4f}"
                        .format(it, multi_step, param_loss.item(), y_loss.item()))

            # Backbone Module
            if condition:
                starting_params = expert_trajectory[start_epoch].copy()
                student_params = torch.cat([p.data.reshape(-1).to(args.device) for p in starting_params], 0).requires_grad_(True)
                
                if args.framework in ['MTT', 'DATM', 'FTD', 'PP', 'TESLA']:
                    MTT(args, GCond, syn_X_total, syn_Y_total, student_net, student_params, target_params, param_dist, criterion)
                elif args.framework == 'DC':
                    DC(args, GCond, syn_X_total, syn_Y_total, student_net, student_params, data_loaders, criterion)
                elif args.framework == 'KIP':
                    KIP(args, GCond, syn_X_total, syn_Y_total, data_loaders)
                elif args.framework == 'FRePo':
                    FRePo(args, GCond, syn_X_total, syn_Y_total, student_net, student_params, data_loaders, criterion)
                elif args.framework == 'DM':
                    DM(args, GCond, syn_X_total, student_net, student_params, data_loaders)
                elif args.framework == 'IDM':
                    IDM(args, GCond, syn_X_total, student_net, student_params, data_loaders, Q_model_pool, Q_max_len)
                else:
                    raise(NotImplementedError)
                
            # CondTSF
            if not condition:
                model_index = np.random.randint(8)
                expert_dir = f'./buffers/{args.dataset}/DLinear/replay_buffer_{model_index}.pt'
                del buffer
                buffer = torch.load(expert_dir)
                expert_trajectory = buffer[0]
                target_params = expert_trajectory[-1].copy()
                target_params = torch.cat([p.data.reshape(-1).to(args.device) for p in target_params], 0)
                
                student_net = STmodels.get_network('DLinear').to(args.device)
                student_net = ReparamModule(student_net)
                student_net.eval()
                
                with torch.no_grad():
                    target_Y = student_net(syn_X_total.permute(0, 2, 1, 3), flat_param=target_params)
            
                begin = input_len
                cnt = 0
                while cnt < target_Y.shape[0]:
                    GCond.feat_syn[:, 0, begin:begin+output_len] = (1 - args.beta) * GCond.feat_syn[:, 0, begin:begin+output_len] + args.beta * target_Y[cnt]
                    cnt += 1
                    begin += inter_step
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='ETTh1', help='dataset')
    parser.add_argument('--model', type=str, default='DLinear', help='model')
    parser.add_argument('--Iteration', type=int, default=300, help='how many distillation steps to perform')
    parser.add_argument('--eval_it', type=int, default=50, help='how often to evaluate')
    parser.add_argument('--lr_teacher', type=float, default=1e-4, help='initialization for synthetic learning rate')
    parser.add_argument('--lr_init', type=float, default=3e-4, help='initialization for synthetic learning rate')
    parser.add_argument('--lr_feat', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")
    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
    parser.add_argument('--epoch_eval_train', type=int, default=250, help='epochs to train a model with synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=4, help='max epoch we can start at')
    parser.add_argument('--expert_epochs', type=int, default=2, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--batch_real', type=int, default=64, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=64, help='batch size for synthetic data')
    parser.add_argument('--batch_train', type=int, default=64, help='batch size for training networks')
    parser.add_argument('--net_mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--framework',type = str, default='MTT', help='name of backbone framework')
    parser.add_argument('--multistep', type=int, default=2, help='times of updating on one trajectory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index')
    parser.add_argument('--alpha', type=float, default=0.2, help='soft label ratio')
    parser.add_argument('--cond_gap', type=int, default=3, help='Epoch gap of using CondTSF')
    parser.add_argument('--beta', type=float, default=0.01, help='CondTSF addtive ratio')
    parser.add_argument('--condensing_ratio', type=str, default='oneshot', help='Condensing ratio, can be choosed from (oneshot, standard, 3-standard)')
    args = parser.parse_args()

    main(args)