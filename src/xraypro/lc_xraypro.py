import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil

import glob
import h5py
import random

#from keras.layers.merge import add
import matplotlib.pyplot as plt
import os
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, Tensor
from sklearn.metrics import mean_absolute_error

from xraypro.transformer_precursor.dataset_modded import MOF_ID_Dataset
from xraypro.transformer_precursor.tokenizer.mof_tokenizer import MOFTokenizer
import csv
import yaml
from xraypro.transformer_precursor.model.utils import *

from xraypro.transformer_precursor.transformer import PositionalEncoding
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

from xraypro.xraypro import loadModel
from xraypro.setGen import xraypro_loaders

class ft_learning_curve():
    def __init__(self, config, config_data, train_loader, test_loader, SEED = 0):
        self.config = config
        self.config_data = config_data
        self.model = loadModel(config = self.config).regressionMode()

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.regression_head.parameters(), lr = 0.01)
        self.optimizer_t = optim.Adam(self.model.model.parameters(), lr = 0.00005)

        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        
        # Paths needed
        self.directory_to_pxrd = self.config_data['path']['path_to_pxrd']
        self.directory_to_precursors = self.config_data['path']['path_to_precursor']
        self.path_to_csv = self.config_data['path']['path_to_csv']

        # declare other configurations needed
        self.label = self.config_data['path']['label']
        self.test_ratio = self.config_data['test_ratio']
        self.valid_ratio = self.config_data['valid_ratio']
        self.batch_size = self.config_data['batch_size']
        self.model_save_path = self.config_data['model_save_path']

        self.train_loader, self.val_loader, self.test_loader = train_loader, test_loader, test_loader
        
        print("Finished loading training, val and test sets!")

    def train(self):
        loss_history, val_history, srcc_val_history = [], [], []
        num_epoch = 100
        best_valid_loss = np.inf

        self.model.train()

        for epoch_counter in range(num_epoch):
            loss_temp = []
            for bn, (input1, input2, target) in enumerate(self.train_loader):
                if torch.cuda.is_available():
                    input_var_1 = input1.to(self.device)
                    input_var_2 = input2.unsqueeze(1).to(self.device)
                    target_var = Variable(target.to(self.device, non_blocking = True))
                    self.model = self.model.to(self.device)

                else:
                    input_var_1 = input1.to(self.device)
                    input_var_2 = input2.unsqueeze(1).to(self.device)
                    target_var = Variable(target)

                target_var = target_var.reshape(-1, 1)
                output = self.model(input_var_2, input_var_1)
                output = output.reshape(-1, 1)

                loss = self.criterion(output, target_var)

                self.optimizer.zero_grad()
                self.optimizer_t.zero_grad()

                loss.backward()

                self.optimizer.step()
                self.optimizer_t.step()

                loss_temp.append(loss.item())
            
            loss_history.append(np.mean(loss_temp))

            val_temp, srcc_val_temp = [], []
            self.model.eval()

            with torch.no_grad():
                for bn, (input1, input2, target) in enumerate(self.val_loader):
                    if torch.cuda.is_available():
                        input_var_1 = input1.to(self.device)
                        input_var_2 = input2.unsqueeze(1).to(self.device)
                        target_var = Variable(target.to(self.device, non_blocking = True))
                        self.model = self.model.to(self.device)

                    else:
                        input_var_1 = input1.to(self.device)
                        input_var_2 = input2.unsqueeze(1).to(self.device)
                        target_var = Variable(target)

                    target_var = target_var.reshape(-1, 1)
                    output = self.model(input_var_2, input_var_1)
                    output = output.reshape(-1, 1)

                    loss_val = self.criterion(output, target_var)
                    val_temp.append(loss_val.item())
                    srcc_val_temp.append(scipy.stats.spearmanr(output.cpu().numpy(), target_var.cpu().numpy())[0])
                
                if np.mean(val_temp) < best_valid_loss:
                    best_valid_loss = np.mean(val_temp)
                    torch.save(self.model.state_dict(), self.model_save_path)
                
                srcc_val_history.append(np.mean(srcc_val_temp))
                val_history.append(np.mean(val_temp))

                if epoch_counter % 1 == 0:
                    print(f'Epoch: {epoch_counter + 1}, Loss: {loss_history[-1]}, Val Loss: {val_history[-1]}, Val SRCC: {srcc_val_history[-1]}')
        
    
    def predict(self):
        self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.eval()

        predictions_test, actual_test = [], []

        for bn, (input1, input2, target) in enumerate(self.test_loader):
            input2 = input2.unsqueeze(1).to(self.device)
            input1 = input1.to(self.device)

            output = self.model(input2, input1)

            for i, j in zip(output.cpu().detach().numpy().flatten(), target.cpu().detach().numpy().flatten()):
                predictions_test.append(i)
                actual_test.append(j)
            
        print(f"Test MAE: {mean_absolute_error(actual_test, predictions_test)}, Test SRCC: {scipy.stats.spearmanr(actual_test, predictions_test)[0]}")
        return predictions_test, actual_test


class sc_learning_curve():
    def __init__(self, config, config_data, train_loader, test_loader, SEED = 0):
        self.config = config
        self.config_data = config_data
        self.model = loadModel(config = self.config, mode = 'None').regressionMode()

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.regression_head.parameters(), lr = 0.01)
        self.optimizer_t = optim.Adam(self.model.model.parameters(), lr = 0.00005)

        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        
        # Paths needed
        self.directory_to_pxrd = self.config_data['path']['path_to_pxrd']
        self.directory_to_precursors = self.config_data['path']['path_to_precursor']
        self.path_to_csv = self.config_data['path']['path_to_csv']

        # declare other configurations needed
        self.label = self.config_data['path']['label']
        self.test_ratio = self.config_data['test_ratio']
        self.valid_ratio = self.config_data['valid_ratio']
        self.batch_size = self.config_data['batch_size']
        self.model_save_path = self.config_data['model_save_path']

        self.train_loader, self.val_loader, self.test_loader = train_loader, test_loader, test_loader
        
        print("Finished loading training, val and test sets!")

    def train(self):
        loss_history, val_history, srcc_val_history = [], [], []
        num_epoch = 100
        best_valid_loss = np.inf

        self.model.train()

        for epoch_counter in range(num_epoch):
            loss_temp = []
            for bn, (input1, input2, target) in enumerate(self.train_loader):
                if torch.cuda.is_available():
                    input_var_1 = input1.to(self.device)
                    input_var_2 = input2.unsqueeze(1).to(self.device)
                    target_var = Variable(target.to(self.device, non_blocking = True))
                    self.model = self.model.to(self.device)

                else:
                    input_var_1 = input1.to(self.device)
                    input_var_2 = input2.unsqueeze(1).to(self.device)
                    target_var = Variable(target)

                target_var = target_var.reshape(-1, 1)
                output = self.model(input_var_2, input_var_1)
                output = output.reshape(-1, 1)

                loss = self.criterion(output, target_var)

                self.optimizer.zero_grad()
                self.optimizer_t.zero_grad()

                loss.backward()

                self.optimizer.step()
                self.optimizer_t.step()

                loss_temp.append(loss.item())
            
            loss_history.append(np.mean(loss_temp))

            val_temp, srcc_val_temp = [], []
            self.model.eval()

            with torch.no_grad():
                for bn, (input1, input2, target) in enumerate(self.val_loader):
                    if torch.cuda.is_available():
                        input_var_1 = input1.to(self.device)
                        input_var_2 = input2.unsqueeze(1).to(self.device)
                        target_var = Variable(target.to(self.device, non_blocking = True))
                        self.model = self.model.to(self.device)

                    else:
                        input_var_1 = input1.to(self.device)
                        input_var_2 = input2.unsqueeze(1).to(self.device)
                        target_var = Variable(target)

                    target_var = target_var.reshape(-1, 1)
                    output = self.model(input_var_2, input_var_1)
                    output = output.reshape(-1, 1)

                    loss_val = self.criterion(output, target_var)
                    val_temp.append(loss_val.item())
                    srcc_val_temp.append(scipy.stats.spearmanr(output.cpu().numpy(), target_var.cpu().numpy())[0])
                
                if np.mean(val_temp) < best_valid_loss:
                    best_valid_loss = np.mean(val_temp)
                    torch.save(self.model.state_dict(), self.model_save_path)
                
                srcc_val_history.append(np.mean(srcc_val_temp))
                val_history.append(np.mean(val_temp))

                if epoch_counter % 1 == 0:
                    print(f'Epoch: {epoch_counter + 1}, Loss: {loss_history[-1]}, Val Loss: {val_history[-1]}, Val SRCC: {srcc_val_history[-1]}')
        
    
    def predict(self):
        self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.eval()

        predictions_test, actual_test = [], []

        for bn, (input1, input2, target) in enumerate(self.test_loader):
            input2 = input2.unsqueeze(1).to(self.device)
            input1 = input1.to(self.device)

            output = self.model(input2, input1)

            for i, j in zip(output.cpu().detach().numpy().flatten(), target.cpu().detach().numpy().flatten()):
                predictions_test.append(i)
                actual_test.append(j)
            
        print(f"Test MAE: {mean_absolute_error(actual_test, predictions_test)}, Test SRCC: {scipy.stats.spearmanr(actual_test, predictions_test)[0]}")
        return predictions_test, actual_test