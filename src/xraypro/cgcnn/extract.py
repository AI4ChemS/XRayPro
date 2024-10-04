#from AtomJSON import AtomCustomJSONInitializer, GaussianDistance
from pymatgen.core.structure import Structure
import torch
import numpy as np
import torch
import warnings
import os
import json

class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.
    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}


    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]


    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}


    def state_dict(self):
        return self._embedding


    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class  AtomCustomJSONInitializer (AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.
    Parameters
    ----------
    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------
        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var


    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array
        Parameters
        ----------
        distance: np.array shape n-d array
          A distance matrix of any shape
        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


def collate_pool_mod(topology_list):
    """
    Collate a list of topology strings (by batch_size) and returns their corresponding properties
    [t1, t2, ..., tN] -> (batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, crystal_atom_idx)
    ||output(1)|| = ||output(2)|| = ||output(3)|| = ||output(4)|| = batch_size

    Details in parameters are found here: https://github.com/zcao0420/MOFormer/blob/a140146bcd0809c7f3afa048aa824432e2e14b60/dataset/dataset_multiview.py
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [] ,[]
    crystal_atom_idx, batch_topology = [], []
    base_idx = 0

    #it would help to map tN -> (atom_fea, nbr_fea, nbr_fea_idx)
    top_features = dict()

    for top in topology_list:
        #cifName = top + '.cif'
        cifName = top
        #print('cifName : %s' % cifName, flush = True)
        warnings.filterwarnings("ignore")

        directory = f'cif/{cifName}.cif'
        #top_features[cifName] = extractFeaturesCGCNN(directory).features()

        try:
            top_features[cifName] = extractFeaturesCGCNN(directory).features() #should give tuple of (atom_fea, nbr_fea, nbr_fea_idx)
        except:
            warningMessage = "Problem with " + cifName
            print(warningMessage)
            print('\n')
    
    for topology, (atom_fea, nbr_fea, nbr_fea_idx) in zip(list(top_features.keys()), list(top_features.values())):
        n_i = atom_fea.shape[0] #number of atoms for this crystal structure
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)
        batch_topology.append(topology)
        base_idx += n_i
    
    return (torch.cat(batch_atom_fea, dim = 0),
            torch.cat(batch_nbr_fea, dim = 0),
            torch.cat(batch_nbr_fea_idx, dim = 0),
            crystal_atom_idx),\
            batch_topology

class extractFeaturesCGCNN():
    def __init__(self, cif, radius = 8, max_num_nbr = 12, dmin = 0, step = 0.2, random_seed = 123):
        self.cif = cif #topSymbol.cif
        self.AtomCustom = AtomCustomJSONInitializer
        self.crystal = Structure.from_file(self.cif)
        self.radius = radius
        self.max_num_nbr = max_num_nbr
        self.dmin = dmin
        self.step = step
        self.random_seed = random_seed
        self.atom_init_file = 'BW20K_code/CGCNN/atom_init.json'
    
    def AtomFeature(self):
        crys = self.crystal.copy()
        atom_fea = np.vstack([self.AtomCustom(self.atom_init_file).get_atom_fea(crys[i].specie.number)
                               for  i  in  range ( len ( crys ))])

        atom_fea = torch.Tensor(atom_fea)
        return atom_fea
    
    def allNeighbours(self):
        all_nbrs = self.crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        return all_nbrs
    
    def features(self):
        nbr_fea_idx , nbr_fea  = [], []
        all_nbrs = extractFeaturesCGCNN(self.cif).allNeighbours()
        atom_fea = extractFeaturesCGCNN(self.cif).AtomFeature()

        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                                'If it happens frequently, consider increase '
                                'radius.'.format(self.cif))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                    [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                                [self.radius + 1.] * (self.max_num_nbr -
                                                        len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)

        selfGDF = GaussianDistance(dmin = self.dmin, dmax = self.radius, step = self.step) #GaussianDistance defined previously (from AtomJSON.py)

        nbr_fea = selfGDF.expand(nbr_fea)
        #atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)

        return (atom_fea, nbr_fea, nbr_fea_idx)
    
    def featureLengths(self):
        atom_fea, nbr_fea, nbr_fea_idx = extractFeaturesCGCNN(self.cif).features()
        orig_atom_fea_len, nbr_fea_len = atom_fea.shape[1], nbr_fea.shape[2]

        return orig_atom_fea_len, nbr_fea_len
    