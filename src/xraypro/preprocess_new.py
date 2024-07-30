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

#from models.MOFormer_modded.transformer import Transformer, TransformerRegressor
from xraypro.MOFormer_modded.dataset_modded import MOF_ID_Dataset
from xraypro.MOFormer_modded.tokenizer.mof_tokenizer import MOFTokenizer
import csv
import yaml
from xraypro.MOFormer_modded.model.utils import *
from xraypro.gaussian import transformPXRD
import pickle

"""
This file helps with preprocessing the PXRD patterns and their precursors to the desired input into the model.
Format of input is usually a dictionary with format {ID: [2theta, intensity, label]}, where ID is the MOF's ID.
The default 2theta bounds are set at (0, 25), but you can edit this file to allow higher/lower angle domain.
"""

class preprocessPXRD():
    def __init__(self, dict_input, directory_mofid, pickle_file = 'uptake_high_p.pickle', two_theta_bounds = (0, 25)):
        self.xrd_label = dict_input
        self.mofid_directory = directory_mofid
        self.two_theta_bounds = two_theta_bounds
        self.pickle_file = pickle_file

    def normalizePXRD(self):
        core_xrd_uptake = dict()

        file_path = self.pickle_file
        if os.path.exists(file_path):
            print("File found. Loading data...")
            with open(file_path, 'rb') as handle:
                core_xrd_uptake = pickle.load(handle)
        else:
            print("File not found. Transforming PXRDs...")
            for id in list(self.xrd_label.keys()):
                x_transformed, y_transformed = transformPXRD(self.xrd_label[id], two_theta_bound = self.two_theta_bounds)
                core_xrd_uptake[id] = [y_transformed, self.xrd_label[id][1]]
        
        return core_xrd_uptake
    
    def MOFid_to_SMILES(self):
        #get MOFids generated
        
        inorg_org = {}

        availableIDs = list(self.normalizePXRD().keys())
        for id in availableIDs:
            try:
                file_path = f'{self.mofid_directory}/{id}.txt'
                f = open(file_path, 'r')
                mofid = f.read().split(' MOFid-v1')[0]
                inorg_org[id] = mofid
            except:
                pass

            #get MOFids
        if len(inorg_org) == 0:
            try:
                file_path = 'core2019/core_mofid.smi'
                inorg_org = dict()

                with open(file_path, 'r') as file:
                    for line in file:
                        mofid_string = line.strip()
                        chemistry_string, cif_name = mofid_string.split(' MOFid-v1')[0], mofid_string.split(';')[1] #gets inorganic and organic portions of MOFid
                        inorg_org[cif_name] = chemistry_string
            except:
                pass

        return inorg_org
    
    def createLoader(self, test_ratio = 0.15, batch_size = 32, randomSeed = 0):
        inorg_org = self.MOFid_to_SMILES()
        core_xrd_uptake = self.normalizePXRD()

        ID_intersect = list(set(list(inorg_org.keys())).intersection(set(list(core_xrd_uptake.keys())))) #now I have IDs that XRD and MOFids both share

        new_d = {'XRD' : [],
                'MOFid' : [],
                'Label' : []
                }

        for id in ID_intersect:
            new_d['XRD'].append(core_xrd_uptake[id][0])
            new_d['Label'].append(core_xrd_uptake[id][1])
            new_d['MOFid'].append(inorg_org[id])
        
        new_df = pd.DataFrame(data = new_d)

        #filter '*' SMILES
        new_df = new_df[new_df['MOFid'] != '*']
        
        __file__ = 'xraypro.py'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        vocab_path = os.path.abspath(os.path.join(current_dir, '..', 'src', 'xraypro', 'MOFormer_modded', 'tokenizer', 'vocab_full.txt'))
        yaml_path = os.path.abspath(os.path.join(current_dir, '..', 'src', 'xraypro', 'MOFormer_modded', 'config_ft_transformer.yaml'))

        tokenizer = MOFTokenizer(vocab_path)
        config = yaml.load(open(yaml_path, "r"), Loader=yaml.FullLoader)
        config['dataloader']['randomSeed'] = 0

        def split_data(data, test_ratio, use_ratio=1, randomSeed = randomSeed):
            total_size = len(data)
            train_ratio = 1 - test_ratio
            indices = list(range(total_size))
            print("The random seed is: ", randomSeed)
            np.random.seed(randomSeed)
            np.random.shuffle(indices)
            train_size = int(train_ratio * total_size)
            test_size = int(test_ratio * total_size)
            print('Total size: {}, Train size: {}, Test size: {}'.format(total_size,
            train_size, test_size
            ))

            train_idx, test_idx = indices[:train_size], indices[-test_size-1:]
            return data[train_idx], data[test_idx]

        data = new_df.to_numpy()

        train_data, test_data = split_data(
            data, test_ratio = test_ratio, 
            randomSeed= randomSeed
        )

        train_dataset = MOF_ID_Dataset(data = train_data, tokenizer = tokenizer)
        test_dataset = MOF_ID_Dataset(data = test_data, tokenizer = tokenizer)

        train_loader = DataLoader(
                        train_dataset, batch_size=batch_size, shuffle = True, drop_last=True
                    )

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = True, drop_last=True
                            )
        
        return train_loader, test_loader
