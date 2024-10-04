import numpy as np
import pandas as pd
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
import pickle
import csv
import yaml

from xraypro.MOFormer_modded.transformer import Transformer, TransformerRegressor
from xraypro.MOFormer_modded.dataset_modded import MOF_ID_Dataset
from xraypro.MOFormer_modded.tokenizer.mof_tokenizer import MOFTokenizer
from xraypro.MOFormer_modded.model.utils import *
from xraypro.cgcnn.extract import extractFeaturesCGCNN
from xraypro.cgcnn.extract import collate_pool_mod
from xraypro.cgcnn.cgcnn_pretrain import CrystalGraphConvNet
from xraypro.MOFormer_modded.transformer import PositionalEncoding
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from SSL.barlow_twins import BarlowTwinsLoss

tokenizer = MOFTokenizer("xraypro/MOFormer_modded/tokenizer/vocab_full.txt")
config = yaml.load(open("xraypro/MOFormer_modded/config_ft_transformer.yaml", "r"), Loader=yaml.FullLoader)
config['dataloader']['randomSeed'] = 0

if torch.cuda.is_available() and config['gpu'] != 'cpu':
    device = config['gpu']
    torch.cuda.set_device(device)
    config['cuda'] = True

else:
    device = 'cpu'
    config['cuda'] = False
print("Running on:", device)

batch_size = 64
class TransformerXRD(nn.Module):
    def __init__(self, input_features=8192, seq_length=batch_size, transformer_heads=8, transformer_ff_dim=512, mlp_hidden_dim=256):
        super(TransformerXRD, self).__init__()
        
        # Project input features to sequence length dimension
        self.input_proj = nn.Linear(input_features, seq_length)
        
        # Define a single Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=seq_length, nhead=transformer_heads, dim_feedforward=transformer_ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # MLP for regression output
        self.mlp = nn.Sequential(
            nn.Linear(seq_length, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)  # Output layer for regression
        )
        self.mlp_manip = nn.Sequential(
            nn.Linear(seq_length, mlp_hidden_dim)
        )
        
    def forward(self, x):
        # Project input features to sequence length dimension
        x_proj = self.input_proj(x)
        x_proj = x_proj.transpose(0, 1)
        
        # Pass through the Transformer encoder
        transformer_output = self.transformer_encoder(x_proj)
        
        # Revert to the original format (N, S) for the MLP
        transformer_output = transformer_output.transpose(0, 1)
        #print("Transformer output shape : {}".format(transformer_output.shape))
        
        # Pass through the MLP
        #output = self.mlp(transformer_output)
        
        # Squeeze the output to remove the extra dimension if the output dimension is 1 -> else, consider alternative for multi-output regression tasks; maybe make conditional?
        return transformer_output

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
        # initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        nn.init.xavier_normal_(self.token_encoder.weight)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.token_encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output[:, 0:1, :] #this was added in by me

        return output.squeeze(dim = 1) #this was added in by me
    
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
            nn.Linear(10, 1),
            nn.ReLU()
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

with open('newSets/ssl_ds_graph.pickle', 'rb') as handle:
    ssl_ds_graph = pickle.load(handle)

with open('newSets/ssl_xrd.pickle', 'rb') as handle:
    ssl_xrd = pickle.load(handle)

with open('newSets/ssl_smiles.pickle', 'rb') as handle:
    ssl_smiles = pickle.load(handle)

data_size = len(ssl_xrd) #should give 8.5k entries for CoRE-2019
split_ratio = 0.95 #train set %
train_size = int(split_ratio * data_size)

ssl_xrd_train, ssl_smiles_train, ssl_graph_train = ssl_xrd[:train_size], ssl_smiles[:train_size], ssl_ds_graph[:train_size]
ssl_xrd_val, ssl_smiles_val, ssl_graph_val = ssl_xrd[train_size-1::], ssl_smiles[train_size-1::], ssl_ds_graph[train_size-1::]

model = UnifiedTransformer(config=config).to(device)

### ESTABLISH CGCNN
orig_atom_fea_len, nbr_fea_len = extractFeaturesCGCNN('cif/str_m1_o1_o1_pcu_sym.15.cif').featureLengths()

config_crystal = yaml.load(open("models/CGCNN/config_ft_cgcnn.yaml", "r"), Loader=yaml.FullLoader) #configurations for CGCNN

config_crystal['model']['orig_atom_fea_len'] = orig_atom_fea_len
config_crystal['model']['nbr_fea_len'] = nbr_fea_len

model_g = CrystalGraphConvNet(**config_crystal['model']).to(device)


loss = BarlowTwinsLoss(device = device, batch_size = 64, embed_size = 512, lambd = 0.00051) #same parameters as in moformer

optimizer_g = optim.Adam(model_g.parameters(), lr = 0.00005)
optimizer_t = optim.Adam(model.parameters(), lr = 0.00005)


### PRETRAINING LOOP
num_epoch = 100
loss_history = []
val_history = []
n_iter = 0
valid_n_iter = 0
best_valid_loss = np.inf
norm = True

for epoch in range(num_epoch):
    model_g.train()
    model.train()

    print(f'Epoch : {epoch + 1}')
    loss_ensemble = []
    for bn, (graph, xrd, smiles) in enumerate(zip(ssl_graph_train, ssl_xrd_train, ssl_smiles_train)):
        input_graph_1 = (Variable(graph[0]).to(device),
                        Variable(graph[1]).to(device),
                        Variable(graph[2]).to(device),
                        [crys_idx.to(device) for crys_idx in graph[3]])
        
        xrd = torch.tensor(xrd, dtype = torch.float).unsqueeze(1).to(device)
        smiles = torch.from_numpy(smiles).to(device)
        
        z_a = model_g(*input_graph_1) #embedding from cgcnn
        z_b = model(xrd, smiles) #embedding from concat. model

        if norm == True:
            z_a_norm = (z_a - torch.mean(z_a, axis = 0))/torch.std(z_a, axis = 0)
            z_b_norm = (z_b - torch.mean(z_b, axis = 0))/torch.std(z_b, axis = 0)
        
        else:
            z_a_norm = z_a
            z_b_norm = z_b

        loss_calc = loss(z_a_norm, z_b_norm)
        loss_ensemble.append(loss_calc.item())

        optimizer_g.zero_grad()
        optimizer_t.zero_grad()

        loss_calc.backward()

        optimizer_g.step()
        optimizer_t.step()
    
    torch.cuda.empty_cache()
    val_ensemble = []

    with torch.no_grad():
        model.eval()
        model_g.eval()
        for bn, (graph, xrd, smiles) in enumerate(zip(ssl_graph_val, ssl_xrd_val, ssl_smiles_val)):
            input_graph_1 = (Variable(graph[0]).to(device),
                            Variable(graph[1]).to(device),
                            Variable(graph[2]).to(device),
                            [crys_idx.to(device) for crys_idx in graph[3]])
            
            xrd = torch.tensor(xrd, dtype = torch.float).unsqueeze(1).to(device)
            smiles = torch.from_numpy(smiles).to(device)

            z_a = model_g(*input_graph_1)
            z_b = model(xrd, smiles)

            if norm == True:
                z_a_norm = (z_a - torch.mean(z_a, axis = 0))/torch.std(z_a, axis = 0)
                z_b_norm = (z_b - torch.mean(z_b, axis = 0))/torch.std(z_b, axis = 0)
            else:
                z_a_norm = z_a
                z_b_norm = z_b
            
            valid_loss = loss(z_a_norm, z_b_norm)
            val_ensemble.append(valid_loss.item())

        val_history.append(np.mean(val_ensemble))
        if np.mean(val_ensemble) < best_valid_loss:
            best_valid_loss = np.mean(val_ensemble)
            
            #save the models here?
            torch.save(model.state_dict(), os.path.join(f'SSL/pretrained/cgcnn', 'model_t.pth'))
            torch.save(model_g.state_dict(), os.path.join(f'SSL/pretrained/cgcnn', 'model_g.pth'))

    
    loss_history.append(np.mean(loss_ensemble))
    print(f'Ensembled Loss : {loss_history[-1]}, Val. Ensembled Loss : {val_history[-1]}')
    print('###############################')