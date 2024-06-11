import os
import argparse
import torch
import torch.nn as nn
from utils import *
from torch.utils.data import TensorDataset, DataLoader
from model import STmodels, gcond
import numpy as np
import warnings
import warnings
import datetime
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(args):
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.curr_time = curr_time
    
    if args.framework == 'Random':
        START_INDEX = '0'
        place1 = 'logged_files'
        place2 = 'MTT'
    else:
        START_INDEX = 'best'
        if args.framework == 'CondTSF':
            place1 = 'logged_files'
            place2 = 'MTT'
        else:
            place1 = 'logged_files_baseline'
            place2 = args.framework
        
    exact_data_lst = os.listdir(f'./{place1}/{place2}/{args.dataset}')
    
    save_dir = os.path.join(".", "log_cross_test", args.framework, args.dataset, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir
    
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    log_verbose(args,"Args : {}\n----------------".format(args.__dict__))

    TS = torch.from_numpy(np.load(f'./data/{args.dataset}/{args.dataset}.npy').T).float()
    input_len = 24
    output_len = 24
    
    TS = (TS - torch.unsqueeze(torch.mean(TS, -1), -1)) / torch.unsqueeze(torch.std(TS, -1), -1)
    TS = torch.unsqueeze(TS, 1)
    N, L = TS.shape[0], TS.shape[2]
    args.N, args.L = N, L
    
    devide = {'train':[0,int(L*0.7)], 'val':[int(L*0.7),L], 'test':[int(L*0.7), L]}
    data_devided = {}
    for k,v in devide.items():
        data_devided[k] = {}
        data_devided[k]['X'], data_devided[k]['Y'] = generate_dataset(TS[:,:,v[0]:v[1]], input_len, output_len, 12)
        data_devided[k]['X'], data_devided[k]['Y'] = data_devided[k]['X'].to(args.device), data_devided[k]['Y'].to(args.device)
    
    data_sets, data_loaders = {}, {}
    for k,v in data_devided.items():
        data_sets[k] = TensorDataset(v['X'].detach(),v['Y'].detach())
        log_verbose(args,'{} : X is {}, Y is {}'.format(k, v['X'].shape, v['Y'].shape))
        data_loaders[k] = DataLoader(data_sets[k], batch_size = args.batch_train, shuffle=True)
    
    model_eval_pool = [args.model]
    log_verbose(args,"model_eval_pool : {}".format(model_eval_pool))
    
    data_index = np.random.randint(len(exact_data_lst))
    
    eval_GCond = gcond.GCOND('oneshot', TS[:,:,devide['train'][0]:devide['train'][1]],args, device=args.device)
    eval_GCond.feat_syn.requires_grad = False
    args.teacher = f'./{place1}/{place2}/{args.dataset}/{exact_data_lst[data_index]}'
    syn_feat = torch.from_numpy(torch.load(os.path.join(args.teacher, 'feat_'+START_INDEX)))
    eval_GCond.feat_syn = nn.Parameter(syn_feat.to(args.device))
    eval_GCond.feat_syn.requires_grad = False

    for model_eval in model_eval_pool:
        MSE_test, MAE_test = [], []
        
        for it_eval in range(args.num_eval):
            net_eval = STmodels.get_network(args.model, channel=N).to(args.device)
            _, _, _, test_MSE, test_MAE = evaluate_synset(it_eval=it_eval, net=net_eval, GCond=eval_GCond, input_len=input_len,
                                                                        output_len=output_len, real_data_dict=data_loaders, args=args, use_data='val')
            
            MSE_test.append(test_MSE)
            MAE_test.append(test_MAE)
            
        MSE_test = np.array(MSE_test)
        MAE_test = np.array(MAE_test)
        
        log_verbose(args,"Eval MSE: {:.6f}".format(MSE_test.mean()))
        log_verbose(args,"Eval MAE: {:.6f}".format(MAE_test.mean()))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='ETTh1', help='dataset')
    parser.add_argument('--model', type=str, default='MLP', help='model')
    parser.add_argument('--lr_teacher', type=float, default=1e-4, help='initialization for synthetic learning rate')
    parser.add_argument('--lr_feat', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--net_mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--batch_train', type=int, default=64, help='batch size for training networks')
    parser.add_argument('--framework',type = str, default='CondTSF', help='name of framework')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index')
    args = parser.parse_args()

    main(args)