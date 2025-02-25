import numpy as np
import pandas as pd
import shutil
import pickle
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import json

from xraypro.transformer_precursor.tokenizer.mof_tokenizer import MOFTokenizer
import csv
import yaml
from xraypro.transformer_precursor.model.utils import *
from xraypro.transformer_precursor.dataset_modded import MOF_ID_Dataset
from torch.utils.data import Dataset, DataLoader, random_split
from xraypro.transform_pxrd import transformPXRD
from sklearn.model_selection import train_test_split

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


def process_pxrd(file_id, directory_to_pxrd, two_theta_bound = (0, 40)):
    calc_pxrd = np.load(f'{directory_to_pxrd}/{file_id}.npy')
    y = transformPXRD(calc_pxrd, two_theta_bound)
    return file_id, y

def transform_PXRD_directory(directory_to_pxrd):
    directory = [f.split('.npy')[0] for f in os.listdir(directory_to_pxrd) if f.endswith('.npy')]
    num_workers = multiprocessing.cpu_count()

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.starmap(process_pxrd, [(file_id, directory_to_pxrd) for file_id in directory])
    id_to_pxrd = dict(results)
    return id_to_pxrd

def xraypro_loaders(directory_to_pxrd, directory_to_precursors, directory_to_csv, property, test_ratio, valid_ratio, config_data, batch_size = 32, SEED = 0):
    df = pd.read_csv(directory_to_csv)

    if list(df.columns) != ['MOF', property]:
        raise ValueError(f"DataFrame columns be 'MOF' and {property} only, but got {list(df.columns)}!")
    
    path_to_pxrd = config_data['path']['path_to_pxrd']
    pxrd_path = f'{path_to_pxrd}/id_to_pxrd.pickle'

    if os.path.exists(pxrd_path):
        with open(pxrd_path, 'rb') as handle:
            id_to_pxrd = pickle.load(handle)
        print("Found id_to_pxrd.pickle!")
    
    else:
        print("id_to_pxrd.pickle cannot be found... Processing PXRDs...")
        id_to_pxrd = transform_PXRD_directory(directory_to_pxrd)
        with open(pxrd_path, 'wb') as handle:
            pickle.dump(id_to_pxrd, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"id_to_pxrd has been deposited to {pxrd_path}")

    availableIDs = np.array(list(df['MOF']))
    PXRD_to_label = {}

    for id in availableIDs:
        try:
            label = df[df['MOF'] == id][property].values[0]
            PXRD_to_label[id] = [id_to_pxrd[id], label]
        except:
            pass
    
    inorg_org = {}
    for id in availableIDs:
        try:
            file_path = f'{directory_to_precursors}/{id}.txt'
            f = open(file_path, 'r')
            precursor = f.read().split(' MOFid-v1')[0]
            inorg_org[id] = precursor
        except:
            pass
    
    ID_intersect = list(set(list(inorg_org.keys())).intersection(set(list(PXRD_to_label.keys()))))

    new_d = {'ID' : [], 'XRD' : [],
             'Precursor' : [],
             'Label' : []
             }
    
    for id in ID_intersect:
        new_d['ID'].append(id)
        new_d['XRD'].append(PXRD_to_label[id][0])
        new_d['Label'].append(PXRD_to_label[id][1])
        new_d['Precursor'].append(inorg_org[id])
    
    new_df = pd.DataFrame(data = new_d)
    new_df = new_df[new_df['Precursor'] != '*']

    data = new_df.to_numpy()

    train_data, test_data, val_data = split_data(
        data, test_ratio = test_ratio, valid_ratio = valid_ratio,
        randomSeed = SEED
    )

    train_cifs, test_cifs, val_cifs = train_data[:, 0], test_data[:, 0], val_data[:, 0]
    train_data, test_data, val_data = train_data[:, 1:], test_data[:, 1:], val_data[:, 1:]

    with open('train_cifs.pickle', 'wb') as handle:
        pickle.dump(train_cifs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('test_cifs.pickle', 'wb') as handle:
        pickle.dump(test_cifs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('val_cifs.pickle', 'wb') as handle:
        pickle.dump(val_cifs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    tokenizer = MOFTokenizer("xraypro/transformer_precursor/tokenizer/vocab_full.txt")
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

def loader_cv(directory_to_pxrd, directory_to_precursors, directory_to_csv, property, config_data, cifs, batch_size = 32, SEED = 0):
    """
    Makes a DataLoader based on the list of CIFs provided - useful for k-CV purposes
    """
    df = pd.read_csv(directory_to_csv)
    df = df[df['MOF'].isin(cifs)]

    if list(df.columns) != ['MOF', property]:
        raise ValueError(f"DataFrame columns be 'MOF' and {property} only, but got {list(df.columns)}!")

    path_to_pxrd = config_data['path']['path_to_pxrd']
    pxrd_path = f'{path_to_pxrd}/id_to_pxrd.pickle'

    if os.path.exists(pxrd_path):
        with open(pxrd_path, 'rb') as handle:
            id_to_pxrd = pickle.load(handle)
        print("Found id_to_pxrd.pickle!")
    
    else:
        print("id_to_pxrd.pickle cannot be found... Processing PXRDs...")
        id_to_pxrd = transform_PXRD_directory(directory_to_pxrd)
        with open(pxrd_path, 'wb') as handle:
            pickle.dump(id_to_pxrd, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"id_to_pxrd has been deposited to {pxrd_path}")

    availableIDs = np.array(list(df['MOF']))
    PXRD_to_label = {}

    for id in availableIDs:
        try:
            label = df[df['MOF'] == id][property].values[0]
            PXRD_to_label[id] = [id_to_pxrd[id], label]
        except:
            pass
    
    inorg_org = {}
    for id in availableIDs:
        try:
            file_path = f'{directory_to_precursors}/{id}.txt'
            f = open(file_path, 'r')
            precursor = f.read().split(' MOFid-v1')[0]
            inorg_org[id] = precursor
        except:
            pass
    
    ID_intersect = list(set(list(inorg_org.keys())).intersection(set(list(PXRD_to_label.keys()))))

    new_d = {'XRD' : [],
             'Precursor' : [],
             'Label' : []
             }
    
    for id in ID_intersect:
        new_d['XRD'].append(PXRD_to_label[id][0])
        new_d['Label'].append(PXRD_to_label[id][1])
        new_d['Precursor'].append(inorg_org[id])
    
    new_df = pd.DataFrame(data = new_d)
    new_df = new_df[new_df['Precursor'] != '*']

    data = new_df.to_numpy()

    tokenizer = MOFTokenizer("xraypro/transformer_precursor/tokenizer/vocab_full.txt")
    cif_dataset = MOF_ID_Dataset(data = data, tokenizer = tokenizer)

    loader_wrt_cifs = DataLoader(
                            cif_dataset, batch_size=batch_size, shuffle = True, drop_last=False
                        )
    
    return loader_wrt_cifs

## For people who want to predict directly from inputted data points: construct dataloader like this

def create_loader(directory_to_pxrd, directory_to_precursors, cif_ids, batch_size = 32, SEED = 0):

    pxrd_file_names = [x.split('.npy')[0] for x in os.listdir(directory_to_pxrd)]
    precursor_file_names = [x.split('.txt')[0] for x in os.listdir(directory_to_precursors)]

    # check to see if all PXRDs are in directory_to_pxrd
    if set(cif_ids).issubset(set(pxrd_file_names)) is False:
        raise ValueError(f"In cif_ids, there are MOFs that are not in {directory_to_pxrd}.")
    
    # check to see if all precursors are in directory_to_precursors
    if set(cif_ids).issubset(set(precursor_file_names)) is False:
        raise ValueError(f"In cif_ids, there are MOFs that are not in {directory_to_precursors}.")

    # create folder called 'subset pxrd' and duplicate files mapping with cif_ids in there
    destination_folder = 'subset/subset pxrd'
    destination_folder_prec = 'subset/subset precursor'

    print(f"Creating {destination_folder}")
    os.makedirs(destination_folder, exist_ok = True)

    print(f"Creating {destination_folder_prec}")
    os.makedirs(destination_folder_prec, exist_ok = True)

    for i in cif_ids:
        source_path = os.path.join(directory_to_pxrd, f'{i}.npy')
        destination_path = os.path.join(destination_folder, f'{i}.npy')
        try:
            shutil.copy2(source_path, destination_path)
        except:
            continue

        source_path = os.path.join(directory_to_precursors, f'{i}.txt')
        destination_path = os.path.join(destination_folder_prec, f'{i}.txt')
        try:
            shutil.copy2(source_path, destination_path)
        except:
            continue
    
    pxrd_path = f'{destination_folder}/data.pickle'

    if os.path.exists(pxrd_path):
        with open(pxrd_path, 'rb') as handle:
            id_to_pxrd = pickle.load(handle)
        
        print("Found data.pickle!")
    
    else:
        print("data.pickle cannot be found... Processing PXRDs...")
        id_to_pxrd = transform_PXRD_directory(destination_folder)
        with open(pxrd_path, 'wb') as handle:
            pickle.dump(id_to_pxrd, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        print(f"data.pickle has been deposited to {pxrd_path}")
    
    availableIDs = np.array(cif_ids)
    PXRD_to_label = {}

    for id in availableIDs:
        try:
            label = 1 #we are making prop predictions assuming no ground truth labels, so label does not matter..
            PXRD_to_label[id] = [id_to_pxrd[id], label]
        except:
            pass

    inorg_org = {}
    for id in availableIDs:
        try:
            file_path = f'{destination_folder_prec}/{id}.txt'
            f = open(file_path, 'r')
            precursor = f.read().split(' MOFid-v1')[0]
            inorg_org[id] = precursor
        except:
            pass
    
    ID_intersect = list(set(list(inorg_org.keys())).intersection(set(list(PXRD_to_label.keys()))))

    new_d = {'XRD' : [],
             'Precursor' : [],
             'Label' : []
             }
    
    for id in ID_intersect:
        new_d['XRD'].append(PXRD_to_label[id][0])
        new_d['Label'].append(PXRD_to_label[id][1])
        new_d['Precursor'].append(inorg_org[id])
    
    new_df = pd.DataFrame(data = new_d)
    new_df = new_df[new_df['Precursor'] != '*']

    data = new_df.to_numpy()

    tokenizer = MOFTokenizer("xraypro/transformer_precursor/tokenizer/vocab_full.txt")
    dataset = MOF_ID_Dataset(data = data, tokenizer = tokenizer)

    data_loader = DataLoader(
                            dataset, batch_size=batch_size, shuffle = True, drop_last=False
                        )
    
    return data_loader

########### THIS IS ONLY FOR ASSESSING MODEL ROBUSTNESS ON GAUSSIAN NOISE. FEEL FREE TO CHANGE! #################################
def process_pxrd_gaussian(file_id, directory_to_pxrd, epsilon, two_theta_bound = (0, 40)):
    """
    Adds Gaussian noise to the PXRD intensities and then transforms it for ML model
    """
    calc_pxrd = np.load(f'{directory_to_pxrd}/{file_id}.npy')
    noise = np.abs(np.random.normal(loc = 1, scale = 1, size = len(calc_pxrd[1])))
    noise = epsilon * noise

    calc_pxrd[1] = calc_pxrd[1] + noise

    y = transformPXRD(calc_pxrd, two_theta_bound)
    return file_id, y    

def transform_PXRD_directory_gaussian(directory_to_pxrd, epsilon):
    directory = [f.split('.npy')[0] for f in os.listdir(directory_to_pxrd) if f.endswith('.npy')]
    num_workers = multiprocessing.cpu_count()

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.starmap(process_pxrd_gaussian, [(file_id, directory_to_pxrd, epsilon) for file_id in directory])
    id_to_pxrd = dict(results)
    return id_to_pxrd

def create_loader_noise(directory_to_pxrd, directory_to_precursors, directory_to_csv, property, cif_ids, epsilon = 0.05, batch_size = 32, SEED = 0):
    df = pd.read_csv(directory_to_csv)

    if list(df.columns) != ['MOF', property]:
        raise ValueError(f"DataFrame columns be 'MOF' and {property} only, but got {list(df.columns)}!")
    
    # create folder called 'subset pxrd' and duplicate files mapping with cif_ids in there
    destination_folder = 'results/noise/gaussian/subset/subset pxrd'
    destination_folder_prec = 'results/noise/gaussian/subset/subset precursor'

    print(f"Creating {destination_folder}")
    os.makedirs(destination_folder, exist_ok = True)

    print(f"Creating {destination_folder_prec}")
    os.makedirs(destination_folder_prec, exist_ok = True)

    for i in cif_ids:
        source_path = os.path.join(directory_to_pxrd, f'{i}.npy')
        destination_path = os.path.join(destination_folder, f'{i}.npy')
        try:
            shutil.copy2(source_path, destination_path)
        except:
            continue

        source_path = os.path.join(directory_to_precursors, f'{i}.txt')
        destination_path = os.path.join(destination_folder_prec, f'{i}.txt')
        try:
            shutil.copy2(source_path, destination_path)
        except:
            continue
    
    pxrd_path = f'{destination_folder}/data_{epsilon}.pickle'

    if os.path.exists(pxrd_path):
        with open(pxrd_path, 'rb') as handle:
            id_to_pxrd = pickle.load(handle)
        
        print(f"Found data_{epsilon}.pickle!")
    
    else:
        print("data.pickle cannot be found... Processing PXRDs...")
        id_to_pxrd = transform_PXRD_directory_gaussian(destination_folder, epsilon)
        with open(pxrd_path, 'wb') as handle:
            pickle.dump(id_to_pxrd, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        print(f"data.pickle has been deposited to {pxrd_path}")
    
    availableIDs = np.array(cif_ids)
    PXRD_to_label = {}

    for id in availableIDs:
        try:
            label = df[df['MOF'] == id][property].values[0]
            PXRD_to_label[id] = [id_to_pxrd[id], label]
        except:
            pass
    
    inorg_org = {}
    for id in availableIDs:
        try:
            file_path = f'{destination_folder_prec}/{id}.txt'
            f = open(file_path, 'r')
            precursor = f.read().split(' MOFid-v1')[0]
            inorg_org[id] = precursor
        except:
            pass
    
    ID_intersect = list(set(list(inorg_org.keys())).intersection(set(list(PXRD_to_label.keys()))))

    new_d = {'XRD' : [],
             'Precursor' : [],
             'Label' : []
             }
    
    for id in ID_intersect:
        new_d['XRD'].append(PXRD_to_label[id][0])
        new_d['Label'].append(PXRD_to_label[id][1])
        new_d['Precursor'].append(inorg_org[id])
    
    new_df = pd.DataFrame(data = new_d)
    new_df = new_df[new_df['Precursor'] != '*']

    data = new_df.to_numpy()

    tokenizer = MOFTokenizer("xraypro/transformer_precursor/tokenizer/vocab_full.txt")
    dataset = MOF_ID_Dataset(data = data, tokenizer = tokenizer)

    data_loader = DataLoader(
                            dataset, batch_size=batch_size, shuffle = True, drop_last=True
                        )
    
    return data_loader

########### THIS IS ONLY FOR ASSESSING MODEL ROBUSTNESS ON CHANGING PEAK WIDTH. FEEL FREE TO CHANGE! #################################

def transform_wrt_sigma(calc_pxrd, two_theta_bound = (0, 40), sigma = 0.1):
    """
    Returns 1D array of intensities of shape (9000,) - this is one of the inputs into XRayPro.
    Input: calc_pxrd -> nested array s.t. [[<---2THETA----->], [<------INTENSITIES------>]]
    sigma -> parameter for peak width.. higher sigma means broader peaks (and vice versa) during transformation.
    """
    data_dict = {'2theta' : calc_pxrd[0],
        'intensity' : calc_pxrd[1]
        }

    data = pd.DataFrame(data = data_dict)

    # Define the function to convert data to Gaussian peaks
    def gaussian(x, mu, sigma):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))
    
    total_points = 9000
    # sigma = 0.1  # Narrow width for thin peaks

    x_transformed = np.linspace(two_theta_bound[0], two_theta_bound[1], total_points)
    y_transformed = np.zeros(total_points)

    for index, row in data[data['intensity'] > 0].iterrows():
        y_transformed += gaussian(x_transformed, row['2theta'], sigma) * row['intensity']

    y_transformed = y_transformed / np.max(y_transformed)

    return y_transformed

def process_pxrd_sigma(file_id, directory_to_pxrd, sigma, two_theta_bound = (0, 40)):
    """
    creates process function (for the sake of multiprocessing) where it accepts sigma of gaussian function as parameter (peak width)
    """
    calc_pxrd = np.load(f'{directory_to_pxrd}/{file_id}.npy')

    y = transform_wrt_sigma(calc_pxrd = calc_pxrd, two_theta_bound = two_theta_bound, sigma = sigma)
    return file_id, y

def transform_PXRD_directory_sigma(directory_to_pxrd, sigma):
    directory = [f.split('.npy')[0] for f in os.listdir(directory_to_pxrd) if f.endswith('.npy')]
    num_workers = multiprocessing.cpu_count()

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.starmap(process_pxrd_sigma, [(file_id, directory_to_pxrd, sigma) for file_id in directory])
    id_to_pxrd = dict(results)
    return id_to_pxrd

def create_loader_sigma(directory_to_pxrd, directory_to_precursors, directory_to_csv, property, cif_ids, sigma = 0.1, batch_size = 32, SEED = 0):
    """
    creates DataLoader for altering peak widths w.r.t. sigma
    """
    df = pd.read_csv(directory_to_csv)

    if list(df.columns) != ['MOF', property]:
        raise ValueError(f"DataFrame columns be 'MOF' and {property} only, but got {list(df.columns)}!")
    
    # create folder called 'subset pxrd' and duplicate files mapping with cif_ids in there
    destination_folder = 'results/noise/peak_width/subset/subset pxrd'
    destination_folder_prec = 'results/noise/peak_width/subset/subset precursor'

    print(f"Creating {destination_folder}")
    os.makedirs(destination_folder, exist_ok = True)

    print(f"Creating {destination_folder_prec}")
    os.makedirs(destination_folder_prec, exist_ok = True)

    for i in cif_ids:
        source_path = os.path.join(directory_to_pxrd, f'{i}.npy')
        destination_path = os.path.join(destination_folder, f'{i}.npy')
        try:
            shutil.copy2(source_path, destination_path)
        except:
            continue

        source_path = os.path.join(directory_to_precursors, f'{i}.txt')
        destination_path = os.path.join(destination_folder_prec, f'{i}.txt')
        try:
            shutil.copy2(source_path, destination_path)
        except:
            continue
    
    pxrd_path = f'{destination_folder}/data_{sigma}.pickle'

    if os.path.exists(pxrd_path):
        with open(pxrd_path, 'rb') as handle:
            id_to_pxrd = pickle.load(handle)
        
        print(f"Found data_{sigma}.pickle!")
    
    else:
        print("data.pickle cannot be found... Processing PXRDs...")
        id_to_pxrd = transform_PXRD_directory_sigma(destination_folder, sigma)
        with open(pxrd_path, 'wb') as handle:
            pickle.dump(id_to_pxrd, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        print(f"data.pickle has been deposited to {pxrd_path}")
    
    availableIDs = np.array(cif_ids)
    PXRD_to_label = {}

    for id in availableIDs:
        try:
            label = df[df['MOF'] == id][property].values[0]
            PXRD_to_label[id] = [id_to_pxrd[id], label]
        except:
            pass
    
    inorg_org = {}
    for id in availableIDs:
        try:
            file_path = f'{destination_folder_prec}/{id}.txt'
            f = open(file_path, 'r')
            precursor = f.read().split(' MOFid-v1')[0]
            inorg_org[id] = precursor
        except:
            pass
    
    ID_intersect = list(set(list(inorg_org.keys())).intersection(set(list(PXRD_to_label.keys()))))

    new_d = {'XRD' : [],
             'Precursor' : [],
             'Label' : []
             }
    
    for id in ID_intersect:
        new_d['XRD'].append(PXRD_to_label[id][0])
        new_d['Label'].append(PXRD_to_label[id][1])
        new_d['Precursor'].append(inorg_org[id])
    
    new_df = pd.DataFrame(data = new_d)
    new_df = new_df[new_df['Precursor'] != '*']

    data = new_df.to_numpy()

    tokenizer = MOFTokenizer("xraypro/transformer_precursor/tokenizer/vocab_full.txt")
    dataset = MOF_ID_Dataset(data = data, tokenizer = tokenizer)

    data_loader = DataLoader(
                            dataset, batch_size=batch_size, shuffle = True, drop_last=True
                        )
    
    return data_loader

############## ADDITION OF MOF-INFORMED PXRDS ###############################

def addPXRD(calc_pxrd, ref_pxrd, epsilon, two_theta_bound = (0, 40)):
    """
    adds a reference PXRD (with some epsilon factor) to PXRD Of interest - this is for noise robustness assessment
    """
    data_dict = {'2theta' : calc_pxrd[0],
        'intensity' : calc_pxrd[1]
        }

    data = pd.DataFrame(data = data_dict)

    # Define the function to convert data to Gaussian peaks
    def gaussian(x, mu, sigma):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))
    
    total_points = 9000
    sigma = 0.1  # Narrow width for thin peaks

    x_transformed = np.linspace(two_theta_bound[0], two_theta_bound[1], total_points)
    y_transformed = np.zeros(total_points)

    for index, row in data[data['intensity'] > 0].iterrows():
        y_transformed += gaussian(x_transformed, row['2theta'], sigma) * row['intensity']

    y_transformed = y_transformed / np.max(y_transformed)
    #y_transformed is PXRD of interest but with 9000 pts

    with open('results/noise/ref/refMOF/nuhqoy/PXRD_before.npy', 'wb') as f:
        np.save(f, y_transformed)

    data_dict = {'2theta' : ref_pxrd[0],
                'intensity' : ref_pxrd[1]
                }
    
    data = pd.DataFrame(data = data_dict)

    x_transformed = np.linspace(two_theta_bound[0], two_theta_bound[1], total_points)
    y_transformed_ref = np.zeros(total_points)
    
    for index, row in data[data['intensity'] > 0].iterrows():
        y_transformed_ref += gaussian(x_transformed, row['2theta'], sigma) * row['intensity']
    
    y_transformed_ref = y_transformed_ref / np.max(y_transformed_ref)
    
    #adds PXRD of interest w/ reference PXRD * epsilon
    y_transformed_prime = y_transformed + epsilon * y_transformed_ref
    y_transformed_prime = y_transformed_prime / np.max(y_transformed_prime)

    with open('results/noise/ref/refMOF/nuhqoy/PXRD_after.npy', 'wb') as f:
        np.save(f, y_transformed_prime)

    return y_transformed_prime

def process_pxrd_ref(file_id, directory_to_pxrd, epsilon, two_theta_bound = (0, 40)):
    """
    adds "noise" to PXRD with a reference PXRD (MOF-5)
    """
    calc_pxrd = np.load(f'{directory_to_pxrd}/{file_id}.npy')
    calc_ref = np.load(f'results/noise/ref/refMOF/nuhqoy/NUHQOY_clean.npy')

    y = addPXRD(calc_pxrd = calc_pxrd, ref_pxrd = calc_ref, epsilon = epsilon, two_theta_bound = two_theta_bound)
    return file_id, y

def transform_PXRD_directory_reference(directory_to_pxrd, epsilon):
    directory = [f.split('.npy')[0] for f in os.listdir(directory_to_pxrd) if f.endswith('.npy')]
    num_workers = multiprocessing.cpu_count()

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.starmap(process_pxrd_ref, [(file_id, directory_to_pxrd, epsilon) for file_id in directory])
    id_to_pxrd = dict(results)
    return id_to_pxrd

def create_loader_ref_noise(directory_to_pxrd, directory_to_precursors, directory_to_csv, property, cif_ids, epsilon = 0.05, batch_size = 32, SEED = 0):
    df = pd.read_csv(directory_to_csv)

    if list(df.columns) != ['MOF', property]:
        raise ValueError(f"DataFrame columns be 'MOF' and {property} only, but got {list(df.columns)}!")
    
    # create folder called 'subset pxrd' and duplicate files mapping with cif_ids in there
    destination_folder = 'results/noise/ref/subset/subset pxrd'
    destination_folder_prec = 'results/noise/ref/subset/subset precursor'

    print(f"Creating {destination_folder}")
    os.makedirs(destination_folder, exist_ok = True)

    print(f"Creating {destination_folder_prec}")
    os.makedirs(destination_folder_prec, exist_ok = True)

    for i in cif_ids:
        source_path = os.path.join(directory_to_pxrd, f'{i}.npy')
        destination_path = os.path.join(destination_folder, f'{i}.npy')
        try:
            shutil.copy2(source_path, destination_path)
        except:
            continue

        source_path = os.path.join(directory_to_precursors, f'{i}.txt')
        destination_path = os.path.join(destination_folder_prec, f'{i}.txt')
        try:
            shutil.copy2(source_path, destination_path)
        except:
            continue
    
    pxrd_path = f'{destination_folder}/data_{epsilon}.pickle'

    if os.path.exists(pxrd_path):
        with open(pxrd_path, 'rb') as handle:
            id_to_pxrd = pickle.load(handle)
        
        print(f"Found data_{epsilon}.pickle!")
    
    else:
        print("data.pickle cannot be found... Processing PXRDs...")
        id_to_pxrd = transform_PXRD_directory_reference(destination_folder, epsilon)
        with open(pxrd_path, 'wb') as handle:
            pickle.dump(id_to_pxrd, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        print(f"data.pickle has been deposited to {pxrd_path}")
    
    availableIDs = np.array(cif_ids)
    PXRD_to_label = {}

    for id in availableIDs:
        try:
            label = df[df['MOF'] == id][property].values[0]
            PXRD_to_label[id] = [id_to_pxrd[id], label]
        except:
            pass
    
    inorg_org = {}
    for id in availableIDs:
        try:
            file_path = f'{destination_folder_prec}/{id}.txt'
            f = open(file_path, 'r')
            precursor = f.read().split(' MOFid-v1')[0]
            inorg_org[id] = precursor
        except:
            pass
    
    ID_intersect = list(set(list(inorg_org.keys())).intersection(set(list(PXRD_to_label.keys()))))

    new_d = {'XRD' : [],
             'Precursor' : [],
             'Label' : []
             }
    
    for id in ID_intersect:
        new_d['XRD'].append(PXRD_to_label[id][0])
        new_d['Label'].append(PXRD_to_label[id][1])
        new_d['Precursor'].append(inorg_org[id])
    
    new_df = pd.DataFrame(data = new_d)
    new_df = new_df[new_df['Precursor'] != '*']

    data = new_df.to_numpy()

    tokenizer = MOFTokenizer("xraypro/transformer_precursor/tokenizer/vocab_full.txt")
    dataset = MOF_ID_Dataset(data = data, tokenizer = tokenizer)

    data_loader = DataLoader(
                            dataset, batch_size=batch_size, shuffle = True, drop_last=True
                        )
    
    return data_loader