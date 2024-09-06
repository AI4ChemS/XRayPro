import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch
import torch.optim as optim
import scipy
import pickle
import os
import yaml

def finetune(model, train_loader, val_loader, test_loader, file_path = 'data/CoRE-MOF/ft', save_path = 'ft_uptake_high_pressure.h5', device = 'cuda:0', num_epoch = 100, label = 'CH4 Uptake at 64 bar'):
    __file__ = 'xraypro.py'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.abspath(os.path.join(current_dir, '..', 'src', 'xraypro', 'MOFormer_modded', 'config_ft_transformer.yaml'))

    config = yaml.load(open(yaml_path, "r"), Loader=yaml.FullLoader)

    new_dir_path = os.path.join(os.getcwd(), file_path, label)
    os.makedirs(new_dir_path, exist_ok = True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.regression_head.parameters(), lr = 0.01)
    optimizer_t = optim.Adam(model.model.parameters(), lr = 0.00005)

    n_iter = 0
    best_srcc_valid = 0

    loss_history, val_history, srcc_val_history = [], [], []
    model.train()

    n_iter = 0
    best_valid_loss = np.inf

    loss_history, val_history, srcc_val_history = [], [], []
    model.train()
    num_epoch = 100

    for epoch_counter in range(num_epoch):
        loss_temp = []
        for bn, (input1, input2, target) in enumerate(train_loader):
            if config['cuda']:
                input_var_1 = input1.to(device) #smiles
                input_var_2 = input2.unsqueeze(1).to(device) #pxrd
            else:
                input_var_1 = input1.to(device)
                input_var_2 = input2.unsqueeze(1).to(device)
            
            
            if config['cuda']:
                #target_var = Variable(target_normed.to(device, non_blocking=True)) #experimenting with normalization vs non-norm
                target_var = Variable(target.to(device, non_blocking=True))
            else:
                #target_var = Variable(target_normed)
                target_var = Variable(target)
            
            if config['cuda']:
                model = model.to(device)

            target_var = target_var.reshape(-1, 1)

            # compute output
            output = model(input_var_2, input_var_1)
            output = output.reshape(-1, 1)

            loss = criterion(output, target_var)

            optimizer.zero_grad()
            optimizer_t.zero_grad()

            loss.backward()

            optimizer.step()
            optimizer_t.step()
            n_iter += 1

            loss_temp.append(loss.item())
        
        loss_history.append(np.mean(loss_temp))

        val_temp = []
        srcc_val_temp = []
        model.eval()
        with torch.no_grad():
            for bn, (input1, input2, target) in enumerate(val_loader):
                if config['cuda']:
                    input_var_1 = input1.to(device)
                    input_var_2 = input2.unsqueeze(1).to(device)
                else:
                    input_var_1 = input1.to(device)
                    input_var_2 = input2.unsqueeze(1).to(device)
                
                if config['cuda']:
                    target_var = Variable(target.to(device, non_blocking=True))
                else:
                    target_var = Variable(target)

                target_var = target_var.reshape(-1, 1)
                # compute output
                output = model(input_var_2, input_var_1)
                output = output.reshape(-1, 1)

                loss_val = criterion(output, target_var)
                val_temp.append(loss_val.item())
                srcc_val_temp.append(scipy.stats.spearmanr(output.cpu().numpy(), target_var.cpu().numpy())[0])
        
        if np.mean(val_temp) < best_valid_loss:
            best_valid_loss = np.mean(val_temp)
            torch.save(model.state_dict(), f'{file_path}/{label}/{save_path}')
        
        elif np.mean(srcc_val_temp) == np.nan:
            pass

        srcc_val_history.append(np.mean(srcc_val_temp))
        val_history.append(np.mean(val_temp))

        if epoch_counter % 1 == 0:
            print(f'Epoch: {epoch_counter+1}, Batch: {bn}, Loss: {loss_history[-1]}, Val Loss: {val_history[-1]}, Val SRCC = {srcc_val_history[-1]}')
    
    return model

def runTest(model, test_loader, save_path = 'ft_uptake_high_p.h5', device = 'cuda:0'):
    #test set
    model.load_state_dict(torch.load(save_path))

    model.eval()
    predictions_test = []
    actual_test = []

    for bn, (input1, input2, target) in enumerate(test_loader):
        # compute output
        input2 = input2.unsqueeze(1).to(device)
        input1 = input1.to(device)
        output = model(input2, input1)
        
        for i, j in zip(output.cpu().detach().numpy().flatten(), target.cpu().detach().numpy().flatten()):
            predictions_test.append(i)
            actual_test.append(j)
    
    return predictions_test, actual_test
