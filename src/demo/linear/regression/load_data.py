#! /bin/env/python3
'''Script to load dnn activations and intersubject eeg data''' 
from pathlib import Path
import joblib
import os
import numpy as np

# function from Ale script encoding_model_utils.py with small modifications -
# instead of project dir input dnn_activations dirt
def load_dnn_data(net, n_pca, project_dir):
	"""Loading the DNN activations of training, validation and test data.
	Parameters
	----------
	net : str
			Used DNN net.
	n_pca : int
			PCA downsampling dimensionality of DNN activations.
	project_dir : str
			Directory of the project folder.

	Returns
	-------
	DNN activations of training, validation and test data.
	"""
	import numpy as np
	import os
	print("\n>>> Loading the DDN activations <<<")

	### DNN activations directory ###
	data_dir = "pca_activations/" + net + "/layers-combined/" \
			+ "pca_" + format(n_pca, "05")
	file_name = "pca_fmaps.npy"
	### Loading the DNN activations ###
	activations = np.load(os.path.join(project_dir, data_dir, file_name), \
			allow_pickle=True).item()

	X_train = activations["fmaps_train"]
	X_val = activations["fmaps_val"]
	X_test = activations["fmaps_test"]
	return X_train, X_val, X_test

def load_intersubject_eeg(inp_dir, filenames):
    '''Loads the output of linear_intersubject_generalization_utils.py
    Input:
        inp_dir - str, directory where the matrix with intersubject data is stored
        filenames - list of strings of filenames to load
    Output:
        files - list of loaded files
    '''
    files = []
    path = Path(inp_dir)
    for fl in filenames:
        files.append(joblib.load(path.joinpath(fl)))
    return files
