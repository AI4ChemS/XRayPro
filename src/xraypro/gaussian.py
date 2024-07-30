import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import os
import pickle

"""
Perform Gaussian transformation on PXRD data
"""
def transformPXRD(uptake_dict, two_theta_bound = (0, 25)):
    calc_pxrd = uptake_dict
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

    return x_transformed, y_transformed


def performTransformation(xrd_uptake):
    def split_dataset(data, n_folds):
        fold_size = len(data) // n_folds
        return [data[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]
    
    all_cifs = list(xrd_uptake.keys())

    folds = np.array(split_dataset(all_cifs, 50))
    print(folds.shape)

    folds = folds.tolist()

    theta_bounds = (0, 90)

    def process_fold(fold):
        fold_results = {}
        for id in fold:
            #print(f"Processing {id}")
            x_transformed, y_transformed = transformPXRD(xrd_uptake[id], two_theta_bound=theta_bounds)
            print(f"Done processing {id}")
            #uptake_ = xrd_uptake[id][2]
            fold_results[id] = [y_transformed]
        return fold_results

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_fold, folds))

    cof_info = {}
    for result in results:
        cof_info.update(result)
    print("Processing complete. Data for each CIF has been processed.")

    folder_name = 'Transformed PXRD'
    current_directory = os.getcwd()
    folder_path = os.path.join(current_directory, folder_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    with open(f'{folder_path}/transformed_PXRD.pickle', 'wb') as handle:
        pickle.dump(cof_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Stored the transformed PXRD data in {folder_path} under transformed_PXRD.pickle")

