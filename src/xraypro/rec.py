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
from xraypro.xraypro import Transformer, CNN_PXRD, UnifiedTransformer, _load_pre_trained_weights
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

__file__ = 'xraypro.py'
current_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.abspath(os.path.join(current_dir, '..', 'src', 'xraypro', 'MOFormer_modded', 'config_ft_transformer.yaml'))

config = yaml.load(open(yaml_path, "r"), Loader=yaml.FullLoader)
config['dataloader']['randomSeed'] = 0

class RecommendationSystem(nn.Module):
    def __init__(self, model, mlp_hidden_dim=256, embed_size = 512):
        super(RecommendationSystem, self).__init__()
        
        #initialize model itself
        self.model = model

        #regression head
        self.regression_head = nn.Sequential(
            nn.Linear(embed_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 6)
        )
        
    def forward(self, xrd, smiles):
        model_output = self.model(xrd, smiles)
        output = self.regression_head(model_output)

        return output

if torch.cuda.is_available():
    device = 'cuda:0'
    torch.cuda.set_device(device)

else:
    device = 'cpu'

class ftRecSys():
    def __init__(self, mode = 'cgcnn'):
        self.concat_model = UnifiedTransformer(config).to(device)
        self.model_pre = _load_pre_trained_weights(model = self.concat_model, mode = mode)
    
    def regressionMode(self):
        model = RecommendationSystem(self.model_pre).to(device)
        return model
