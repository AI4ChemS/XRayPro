import numpy as np
import torch
import functools

from CoRE_2019.CGCNN.extract import collate_pool_mod, extractFeaturesCGCNN

class MOF_ID_Dataset_Graph(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch for CGCNN included -> written by Sartaaj Khan'
    def __init__(self, data, tokenizer, use_ratio = 1):

            self.data = data[:int(len(data)*use_ratio)]
            self.graph = self.data[:, 4].astype(str) #get the graph data
            self.xrd = np.vstack(self.data[:, 1])
            self.mofid = self.data[:, 2].astype(str) #used to be [:, 1]
            self.tokens = np.array([tokenizer.encode(i, max_length=512, truncation=True,padding='max_length') for i in self.mofid])
            self.label = self.data[:, 3].astype(float) #used to be [:, 2]

            self.tkenizer = tokenizer

    def __len__(self):
            return len(self.label)
            
    @functools.lru_cache(maxsize=None) 
    def __getitem__(self, index):
            # Load data and get label
            X1 = torch.from_numpy(np.asarray(self.tokens[index])) #SMILES
            X2 = torch.tensor(self.xrd[index], dtype = torch.float) #XRD

            #get graph features
            X3 = self.graph.copy()

            #X = torch.concat((X1, X2))
            #X = torch.from_numpy(np.asarray(self.tokens[index]))
            y = torch.from_numpy(np.asarray(self.label[index]))

            return X1, X2, X3, y.float()