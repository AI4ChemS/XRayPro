import numpy as np
import pandas as pd
import pickle

from xraypro.MOFormer_modded.tokenizer.mof_tokenizer import MOFTokenizer
import csv
import yaml
from xraypro.MOFormer_modded.model.utils import *
from xraypro.MOFormer_modded.dataset_modded import MOF_ID_Dataset
from torch.utils.data import Dataset, DataLoader, random_split

def split_data(data, test_ratio, valid_ratio, use_ratio=1, randomSeed = 0):
    """
    Generates train, test and val. sets. Original source: https://github.com/zcao0420/MOFormer 
    """
    total_size = len(data)
    train_ratio = 1 - valid_ratio - test_ratio
    indices = list(range(total_size))
    print("The random seed is: ", randomSeed)
    np.random.seed(randomSeed)
    np.random.shuffle(indices)
    train_size = int(train_ratio * total_size)
    valid_size = int(valid_ratio * total_size)
    test_size = int(test_ratio * total_size)
    print('Train size: {}, Validation size: {}, Test size: {}'.format(
    train_size, valid_size, test_size
    ))
    train_idx, valid_idx, test_idx = indices[:train_size], indices[-(valid_size + test_size):-test_size], indices[-test_size:]
    return data[train_idx], data[valid_idx], data[test_idx]

def genLoaders(PXRD_to_Label, directory_to_precursors, test_ratio, valid_ratio, batch_size = 32, SEED = 0):
    """
    PXRD_to_Label MUST be in this format: {ID : [1D array of PXRD, Label]}
    directory_to_precursors assumes that you have a folder of saved .txt files of the precursors from getPrecursor.py
    """
    #get MOFids generated
    inorg_org = {}
    availableIDs = PXRD_to_Label.keys()

    for id in availableIDs:
        try:
            file_path = f'{directory_to_precursors}/{id}.txt'
            f = open(file_path, 'r')
            precursor = f.read().split(' MOFid-v1')[0]
            inorg_org[id] = precursor
        except:
            pass
    
    ID_intersect = list(set(list(inorg_org.keys())).intersection(set(list(PXRD_to_Label.keys())))) #now I have IDs that XRD and MOFids both share

    new_d = {'XRD' : [],
            'MOFid' : [],
            'Label' : []
            }

    for id in ID_intersect:
        new_d['XRD'].append(PXRD_to_Label[id][0])
        new_d['Label'].append(PXRD_to_Label[id][1])
        new_d['MOFid'].append(inorg_org[id])
    
    new_df = pd.DataFrame(data = new_d)

    #filter '*' SMILES
    new_df = new_df[new_df['MOFid'] != '*']

    data = new_df.to_numpy()

    train_data, test_data, val_data = split_data(
        data, test_ratio = test_ratio, valid_ratio = valid_ratio,
        randomSeed = SEED
    )


    tokenizer = MOFTokenizer("xraypro/MOFormer_modded/tokenizer/vocab_full.txt")
    train_dataset = MOF_ID_Dataset(data = train_data, tokenizer = tokenizer)
    test_dataset = MOF_ID_Dataset(data = test_data, tokenizer = tokenizer)
    val_dataset = MOF_ID_Dataset(data = val_data, tokenizer=tokenizer)

    train_loader = DataLoader(
                            train_dataset, batch_size=batch_size, shuffle = True, drop_last=True
                        )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = True, drop_last=True
                        )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = True, drop_last=True
                        )
    
    return train_loader, test_loader, val_loader
