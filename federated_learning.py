import collections
import copy
from tqdm import tqdm
import torch

from training import evaluate_model
from diffusion import diffusion_loss

def weighted_averaging(w_list, num_sample_list):
    num_total_samples = sum(num_sample_list)
    keys = w_list[0].keys()
    w_avg = collections.OrderedDict()

    device = w_list[0][list(keys)[0]].device
    
    for k in keys:
        w_avg[k] = torch.zeros(w_list[0][k].size()).to(device)   # Reshape w_avg to match local weights.

    for k in keys:
        for i in range(len(w_list)):
            w_avg[k] += num_sample_list[i] * w_list[i][k]
        w_avg[k] = torch.div(w_avg[k], num_total_samples)
    return w_avg


def local_update_fedavg(glob_model, client_loader, num_local_epochs, optim_args, test_loader=None):

    local_model = copy.deepcopy(glob_model)
    local_model.train()
    device = local_model.device
    optimizer = torch.optim.Adam(local_model.parameters(), **optim_args)
    grad_scaler = torch.cuda.amp.GradScaler()
    
    # Training.
    for epoch in range(num_local_epochs):
        for (x, y) in tqdm(client_loader, desc='epoch {}/{}'.format(epoch + 1, num_local_epochs)):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)

            # Calculate loss.
            loss = diffusion_loss(local_model, x, y)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
    
    # Evaluating.
    train_loss = evaluate_model(local_model, client_loader, diffusion_loss, tqdm_desc='local_train_loss')
    
    # Return update.
    local_update_dict ={
        'local_w': local_model.state_dict(),
        'num_samples': len(client_loader.dataset),
        'train_loss': train_loss,
    }
    
    if test_loader:
        test_loss = evaluate_model(local_model, test_loader, diffusion_loss, tqdm_desc='local_test_loss')
        local_update_dict['test_loss'] = test_loss
    
    return local_update_dict