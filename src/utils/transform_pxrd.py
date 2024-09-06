import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import os
import pickle

from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.diffraction.xrd import XRDCalculator

def transformPXRD(calc_pxrd, two_theta_bound = (0, 25)):
    """
    Returns 1D array of intensities of shape (9000,) - this is one of the inputs into XRayPro.
    Input: calc_pxrd -> nested array s.t. [[<---2THETA----->], [<------INTENSITIES------>]]
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

    return y_transformed

def XRDPattern(directory_to_cif):
    """
    Uses Pymatgen to calculate PXRD pattern. Defaults to returning 2THETA bounds from (0, 90) degrees.
    """
    structure = Structure.from_file(directory_to_cif)
    xrd_calculator = XRDCalculator()
    xrd_pattern = xrd_calculator.get_pattern(structure)
    return xrd_pattern.x.tolist(), xrd_pattern.y.tolist()

def CIF_to_PXRD(directory_to_cif, two_theta_bound = (0, 25)):
    """
    Computes PXRD of one CIF and then transforms it to (9000,) shape, applying Gaussian transformation.
    directory_to_cif: string of directory to CIF of interest.
    """
    x, y = XRDPattern(directory_to_cif)
    calc_pxrd = np.array([x, y])

    y_transformed = transformPXRD(calc_pxrd, two_theta_bound)
    return y_transformed


def performTransformation(xrd_uptake, theta_bounds = (0, 25)):
    def split_dataset(data, n_folds):
        fold_size = len(data) // n_folds
        return [data[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]
    
    all_cifs = list(xrd_uptake.keys())

    folds = np.array(split_dataset(all_cifs, 50))
    print(folds.shape)

    folds = folds.tolist()

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

def expPXRDTransform(directory_to_xy, two_theta_bound = (0, 40)):
    """
    Accepts a .xy file (give path to it) and transform PXRD pattern
    """
    data = np.loadtxt(directory_to_xy, skiprows = 1)
    x, y = data[:, 0], data[:, 1]

    pattern = np.array([x, y])
    y_t = transformPXRD(pattern, two_theta_bound=two_theta_bound)

    return y_t