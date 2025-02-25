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

from xraypro.transformer_precursor.dataset_modded import MOF_ID_Dataset
from xraypro.transformer_precursor.tokenizer.mof_tokenizer import MOFTokenizer
import csv
import yaml
from xraypro.transformer_precursor.model.utils import *

from xraypro.transformer_precursor.transformer import PositionalEncoding
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

__file__ = 'xraypro.py'

class Transformer(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.xavier_normal_(self.token_encoder.weight)

    def forward(self, src: Tensor) -> Tensor:
        """
        Modded from: https://pubs.acs.org/doi/10.1021/jacs.2c11420 
        """
        src = self.token_encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output[:, 0:1, :]
        return output.squeeze(dim = 1)
    

class CNN_PXRD(nn.Module):
    """
    CNN that accepts PXRD pattern of dimension (N, 1, 9000) and returns some regression output (N, 1)
    Usage: CNN_PXRD(X) -> returns predictions
    If dim(X) = (N, 9000), do X.unsqueeze(1) and thene input that into model.
    """
    def __init__(self):
        super(CNN_PXRD, self).__init__()

        self.maxpool1 = nn.MaxPool1d(kernel_size=3) # returns (N, 1, 3000)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3) # returns (N, 5, 2998)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=5, out_channels=5, kernel_size=3) # returns (N, 5, 2996)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2) # returns (N, 5, 1498)
        self.conv3 = nn.Conv1d(in_channels=5, out_channels=10, kernel_size=3) # returns (N, 10, 1496)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3) # returns (N, 10, 1494)
        self.relu4 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=2) # returns (N, 10, 747)
        self.conv5 = nn.Conv1d(in_channels=10, out_channels=15, kernel_size=5) # returns (N, 15, 743)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv1d(in_channels=15, out_channels=15, kernel_size=5) # returns (N, 15, 739)
        self.relu6 = nn.ReLU()
        self.maxpool4 = nn.MaxPool1d(kernel_size=3) # returns (N, 15, 246)
        self.conv7 = nn.Conv1d(in_channels=15, out_channels=20, kernel_size=5) # returns (N, 20, 242)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=5) # returns (N, 20, 238)
        self.relu8 = nn.ReLU()
        self.maxpool5 = nn.MaxPool1d(kernel_size=2) # returns (N, 20, 119)
        self.conv9 = nn.Conv1d(in_channels=20, out_channels=30, kernel_size=5) # returns (N, 30, 115)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv1d(in_channels=30, out_channels=30, kernel_size=5) # returns (N, 30, 111)
        self.relu10 = nn.ReLU()
        self.maxpool6 = nn.MaxPool1d(kernel_size=5) # returns (N, 30, 22)
        self.flatten = nn.Flatten() # returns (N, 660)
        self.fc1 = nn.Linear(660, 80) # returns (N, 80)
        self.relu11 = nn.ReLU()
        self.fc2 = nn.Linear(80, 50) # returns (N, 50)
        self.relu12 = nn.ReLU()
        self.fc3 = nn.Linear(50, 10) # returns (N, 10)
        self.relu13 = nn.ReLU()
        self.fc4 = nn.Linear(10, 1) # returns (N, 1)
        self.relu14 = nn.ReLU()

        self.regression_head = nn.Sequential(
            nn.Linear(660, 80),
            nn.ReLU(),
            nn.Linear(80, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

        self.apply(self.weights_init) #need to initialize weights otherwise grad. shoots to infinite

    def forward(self, x):
        x = self.maxpool1(x) # (N, 1, 3000)
        x = self.conv1(x) # (N, 5, 2998)
        x = self.relu1(x)
        x = self.conv2(x) # (N, 5, 2996)
        x = self.relu2(x)
        x = self.maxpool2(x) # (N, 5, 1498)
        x = self.conv3(x) # (N, 10, 1496)
        x = self.relu3(x)
        x = self.conv4(x) # (N, 10, 1494)
        x = self.relu4(x)
        x = self.maxpool3(x) # (N, 10, 747)
        x = self.conv5(x) # (N, 15, 743)
        x = self.relu5(x)
        x = self.conv6(x) # (N, 15, 739)
        x = self.relu6(x)
        x = self.maxpool4(x) # (N, 15, 246)
        x = self.conv7(x) # (N, 20, 242)
        x = self.relu7(x)
        x = self.conv8(x) # (N, 20, 238)
        x = self.relu8(x)
        x = self.maxpool5(x) # (N, 20, 119)
        x = self.conv9(x) # (N, 30, 115)
        x = self.relu9(x)
        x = self.conv10(x) # (N, 30, 111)
        x = self.relu10(x)
        x = self.maxpool6(x) # (N, 30, 22)
        x = self.flatten(x) # (N, 660)
        return x
    
    def weights_init(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class UnifiedTransformer(nn.Module):
    def __init__(self, config, mlp_hidden_dim = 256):
        super(UnifiedTransformer, self).__init__()
        
        #transformer for embedding SMILES
        self.transformer1 = Transformer(**config['Transformer'])
        
        #CNN for embedding PXRD
        self.cnn = CNN_PXRD()

        #projector
        self.proj = nn.Sequential(
            nn.Linear(1172, mlp_hidden_dim),
            nn.Softplus(),
            nn.Linear(mlp_hidden_dim, 512)
        )
                
    def forward(self, xrd, smiles):
        transformer1_output = self.transformer1(smiles) #gets output from SMILES transformer -> shape of (batchSize, 512, 512)
        transformer2_output = self.cnn(xrd) #gets output from XRD transformer -> shape of (batchSize, seq_len)

        concatenated_tensor_corrected = torch.cat((transformer1_output, transformer2_output), dim=1)
        
        proj_out = self.proj(concatenated_tensor_corrected)
        return proj_out

def _load_pre_trained_weights(model, config, mode = 'cgcnn'):
    """
    Taken from this repository: https://github.com/zcao0420/MOFormer/blob/main/finetune_transformer.py
    """
    __file__ = 'xraypro.py'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ssl_path = os.path.abspath(os.path.join('pretrained', 'cgcnn'))
    
    try:
        if mode == 'cgcnn':
            checkpoints_folder = ssl_path
        
        else:
            checkpoints_folder = 'SSL/pretrained/None'

        load_state = torch.load(os.path.join(checkpoints_folder, 'model_t.pth'),  map_location=config['gpu']) 
        model_state = model.state_dict()

        for name, param in load_state.items():
            if name not in model_state:
                i = 2 #some variable
                continue
            else:
                i = 2 #some variable
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            model_state[name].copy_(param)
        print("Loaded pre-trained model with success.")
    
    except FileNotFoundError:
        print("Pre-trained weights not found. Training from scratch.")

    print(os.path.join(checkpoints_folder, 'model_t.pth'))

    return model

class UnifiedTransformer_Regression(nn.Module):
    def __init__(self, model, mlp_hidden_dim=256, embed_size = 512):
        super(UnifiedTransformer_Regression, self).__init__()
        
        #initialize model itself
        self.model = model

        #regression head
        self.regression_head = nn.Sequential(
            nn.Linear(embed_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)
        )
        
    def forward(self, xrd, smiles):
        model_output = self.model(xrd, smiles)
        output = self.regression_head(model_output)

        return output

class UnifiedTransformer_Classification(nn.Module):
    def __init__(self, model, mlp_hidden_dim=256, embed_size = 512, threshold = 0.4):
        super(UnifiedTransformer_Classification, self).__init__()
        
        self.model = model
        self.threshold = threshold

        #classification head
        self.classification_head = nn.Sequential(
            nn.Linear(embed_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, xrd, smiles):
        model_output = self.model(xrd, smiles)
        output = self.classification_head(model_output)

        return output

class loadModel():
    def __init__(self, config, mode = 'cgcnn'):
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            torch.cuda.set_device(self.device)

        else:
            self.device = 'cpu'

        self.concat_model = UnifiedTransformer(config).to(self.device)
        self.model_pre = _load_pre_trained_weights(model = self.concat_model, mode = mode, config = config)
    
    def regressionMode(self):
        model = UnifiedTransformer_Regression(self.model_pre).to(self.device)
        return model
    
    def classificationMode(self):
        model = UnifiedTransformer_Classification(self.model_pre).to(self.device)
        return model

# made class to make model predictions directly from inputted data points; should probably include a class for preprocessing datapoints

class XRayPro():
    def __init__(self, config, path_to_weights, SEED = 0, embedding = False):
        """
        class should be able to take in any DataLoader and output predictions
        """
        self.embedding = embedding
        self.config = config
        self.path_to_weights = path_to_weights
        self.model = loadModel(self.config, mode = 'None').regressionMode()

        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

    def predict(self, loader):
        self.model.load_state_dict(torch.load(self.path_to_weights))
        self.model = self.model.to(self.device)
        self.model.eval()

        predictions = []

        for bn, (input1, input2, _) in enumerate(loader):
            input2 = input2.unsqueeze(1).to(self.device)
            input1 = input1.to(self.device)

            output = self.model(input2, input1)

            for i in output.cpu().detach().numpy().flatten():
                predictions.append(i)
        
        return predictions