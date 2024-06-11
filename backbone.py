import torch
import numpy as np


def MTT(args, GCond, syn_X_total, syn_Y_total, student_net, student_params, target_params, param_dist, criterion):
    G = None
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
        grad = torch.autograd.grad(mse_loss, student_params, create_graph=True, allow_unused=True)[0]
        
        if args.framework == 'TESLA':
            if G == None:
                G = grad
            else:
                G = G + grad
        
        student_params = student_params - args.lr_init * grad
    
    if args.framework != 'TESLA':
        if args.framework == 'PP':
            epsilon = 0.02
            with torch.no_grad():
                for i in range(len(student_params)):
                    if torch.abs(student_params[i] / target_params[i] - 1) < epsilon:
                        student_params[i] = target_params[i]
            
        param_loss = torch.nn.functional.mse_loss(student_params, target_params, reduction="sum")
        param_loss /= param_dist
        loss = param_loss / param_loss.item()
    else:
        l1 = 2 * args.lr_init * (target_params - student_params).T @ grad
        l2 = 2 * (args.lr_init ** 2) * G.T @ grad
        loss = l1 + l2
    
    GCond.optimizer_feat.zero_grad()
    loss.backward()
    GCond.optimizer_feat.step()
    
    
def DC(args, GCond, syn_X_total, syn_Y_total, student_net, student_params, data_loaders, criterion):
    for step in range(args.syn_steps):
        b = args.batch_syn
        perm = torch.randperm(syn_X_total.shape[0])[:b]
        syn_X = syn_X_total[perm,:,:]
        index = np.random.randint(len(data_loaders['train']))
        for i, (X, Y) in enumerate(data_loaders['train']):
            if i == index:
                real_X, real_Y = X, Y
                break  
        
        syn_X = syn_X.permute(0, 2, 1, 3)
        y_pred = student_net(syn_X, flat_param = student_params)
        y_pred_real = student_net(real_X.permute(0, 2, 1, 3), flat_param=student_params)
            
        mse_loss_1 = criterion(y_pred, syn_Y_total[perm,:,:])
        mse_loss_2 = criterion(y_pred_real, real_Y)
        
        grad_1 = torch.autograd.grad(mse_loss_1, student_params, create_graph=True, allow_unused=True)[0]
        grad_2 = torch.autograd.grad(mse_loss_2, student_params, create_graph=True, allow_unused=True)[0]
        
        grad_loss = torch.nn.functional.mse_loss(grad_1, grad_2, reduction="mean")
        GCond.optimizer_feat.zero_grad()
        grad_loss.backward(retain_graph=True)
        GCond.optimizer_feat.step()
        
        student_params = student_params - args.lr_init * grad_1
        
        
def KIP(args, GCond, syn_X_total, syn_Y_total, data_loaders):
    
    def rbf_kernel(x1, x2):
        sigma = 3
        dist = torch.cdist(x1, x2)
        K = torch.exp(-dist / (2 * sigma**2))
        return K
    
    index = np.random.randint(len(data_loaders['train']) - 1)
    for i, (X, Y) in enumerate(data_loaders['train']):
        if i == index:
            real_X, real_Y = X, Y
            break
    size = syn_X_total.shape[0]
    real_X = real_X[:size].reshape(size, -1)
    real_Y = real_Y[:size].reshape(size, -1)
    syn_X = syn_X_total.reshape(size, -1)
    syn_Y = syn_Y_total.reshape(size, -1)
    
    K_ts = rbf_kernel(real_X, syn_X)
    K_ss = rbf_kernel(syn_X, syn_X)
    
    loss = torch.mean(torch.norm(real_Y - K_ts @ torch.inverse(K_ss + torch.eye(size).to(args.device)) @ syn_Y))
    GCond.optimizer_feat.zero_grad()
    loss.backward()
    GCond.optimizer_feat.step()
    
    
def FRePo(args, GCond, syn_X_total, syn_Y_total, student_net, student_params, data_loaders, criterion):
    for step in range(args.syn_steps):
        b = args.batch_syn
        perm = torch.randperm(syn_X_total.shape[0])[:b]
        syn_X = syn_X_total[perm,:,:]
        syn_Y = syn_Y_total[perm]
        index = np.random.randint(len(data_loaders['train']) - 1)
        for i, (X, Y) in enumerate(data_loaders['train']):
            if i == index:
                real_X, real_Y = X, Y
                break
        
        size = syn_X.shape[0]
        real_X = real_X[:size]
        real_Y = real_Y[:size]
        
        syn_X = syn_X.permute(0, 2, 1, 3)
        y_pred = student_net(syn_X, flat_param=student_params)
        y_pred_real = student_net(real_X.permute(0, 2, 1, 3), flat_param=student_params)
        
        y_pred = y_pred.reshape(size, -1)
        y_pred_real = y_pred_real.reshape(size, -1)
        syn_Y = syn_Y.reshape(size, -1)
        real_Y = real_Y.reshape(size, -1)
        
        K_ts = y_pred_real @ y_pred.T
        K_ss = y_pred @ y_pred.T + torch.eye(size).to(args.device)
        
        loss = torch.mean(torch.norm(real_Y - K_ts @ torch.inverse(K_ss) @ syn_Y))
        
        GCond.optimizer_feat.zero_grad()
        loss.backward(retain_graph=True)
        GCond.optimizer_feat.step()
        
        mse_loss = criterion(y_pred, syn_Y)
        grad = torch.autograd.grad(mse_loss, student_params, create_graph=True, allow_unused=True)[0]
        student_params = student_params - args.lr_init * grad
        
        
def DM(args, GCond, syn_X_total, student_net, student_params, data_loaders):
    b = args.batch_syn
    perm = torch.randperm(syn_X_total.shape[0])[:b]
    syn_X = syn_X_total[perm,:,:]
    index = np.random.randint(len(data_loaders['train']))
    for i, (X, Y) in enumerate(data_loaders['train']):
        if i == index:
            real_X, real_Y = X, Y
            break
    syn_X = syn_X.permute(0, 2, 1, 3)
    y_pred = student_net(syn_X, flat_param = student_params)
    y_pred_real = student_net(real_X.permute(0, 2, 1, 3), flat_param=student_params)
        
    y_pred = torch.mean(y_pred, dim=0)
    y_pred_real = torch.mean(y_pred_real, dim=0)
    
    loss = torch.nn.functional.mse_loss(y_pred, y_pred_real, reduction="mean")  
    GCond.optimizer_feat.zero_grad()
    loss.backward()
    GCond.optimizer_feat.step()
    
    
def IDM(args, GCond, syn_X_total, student_net, student_params, data_loaders, Q_model_pool, Q_max_len):
    Q_model_pool.append(student_params)
    if len(Q_model_pool) > Q_max_len:
        Q_model_pool.pop(0)
        
    student_index = np.random.randint(len(Q_model_pool))
    student_params = Q_model_pool[student_index]
    
    b = args.batch_syn
    perm = torch.randperm(syn_X_total.shape[0])[:b]
    syn_X = syn_X_total[perm,:,:]
    index = np.random.randint(len(data_loaders['train']))
    for i, (X, Y) in enumerate(data_loaders['train']):
        if i == index:
            real_X, real_Y = X, Y
            break
    syn_X = syn_X.permute(0, 2, 1, 3)
    y_pred = student_net(syn_X, flat_param=student_params)
    y_pred_real = student_net(real_X.permute(0, 2, 1, 3), flat_param=student_params)
        
    y_pred = torch.mean(y_pred, dim=0)
    y_pred_real = torch.mean(y_pred_real, dim=0)
    
    loss = torch.nn.functional.mse_loss(y_pred, y_pred_real, reduction="mean")  
    GCond.optimizer_feat.zero_grad()
    loss.backward(retain_graph=True)
    GCond.optimizer_feat.step()
                
    for train_epochs in range(args.syn_steps):
        index = np.random.randint(len(data_loaders['train']))
        for i, (X, Y) in enumerate(data_loaders['train']):
            if i == index:
                real_X, real_Y = X, Y
                break
        y_pred_real = student_net(real_X.permute(0, 2, 1, 3), flat_param=student_params)
        loss_train = torch.nn.functional.mse_loss(real_Y, y_pred_real, reduction="mean")
        grad = torch.autograd.grad(loss_train, student_params, create_graph=True, allow_unused=True)[0]
        student_params = student_params - args.lr_init * grad
        
    Q_model_pool[student_index] = student_params