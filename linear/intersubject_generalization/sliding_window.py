#! /bin/env/python3
'''
Script implements intersubject generalization with sliding window over time points.
For each if the time point ranges in the raw data the featurematrix (channels x time) 
is created and intersubject generalizer object is fitted on each of these snippets.
'''
import numpy as np
import joblib 
import copy
import warnings
from pathlib import Path
# directory with feature matrices scripts
import sys
sys.path.append('/home/akitaitsev/code/intersubject_generalization/linear/create_featurematrices')
from create_featurematrix import dataset2feature_matrix
from linear_intersubject_generalization_utils import intersubject_generalizer

__all__ = ['sliding_window']

def sliding_window(generalizer, dataset_train, dataset_val, dataset_test, window_length, \
    hop_length=None):
    '''
    Sliding window intersubject geenralizaion. Generalizer object is fit on time window of
    train dataset and used to predict val and test datasets in the same window.
    
    Note that if feature space of the data can not be divided into integer number of windows,
    the last window will contain all the remainder feautre samples (will be shorter than others).
    
    Inputs:
        generalizer - intersubject generalizer object to be fitted on tarin data
        dataset_train - 4d numpy array of shape (subj, images, channels, times)
        dataset_val - 4d numpy array of shape (subj, images, channels, times)
        dataset_test - 4d numpy array of shape (subj, images, channels, times)
        window_length - int, length of liding window in samples
        hop_length - int, step size of sliding window replacement. If None,
                     hop_length==window_length. Default=None
    Outputs:
        projected_data - list of projected train, val and test data (each of of the entries
                         is a list of (shape n_windows +1))
        backprojected_data -  list of projected train, val and test data
        generalizers - list of generalizer objects of length n_windows
        metadata - dictionary with hop_lenth, window_length and indices used for feature
                    indexing.
    '''
    if hop_length==None:
        hop_length = window_length
    if hop_length != window_length:
        warnings.warn("Hop length is not equal to window length. Current version of the function is "
        "stable only for non-overlapping windows! Check out ind_list to make sure everything works fine!")
    if hop_length > window_length:
        warinings.warn("hop length is larger than window length! There will be unanalized data snippets!")
    if dataset_train.shape[-1] < window_length:
        raise ValueError("Window length is larger than the number of times!")
    if dataset_train.shape[-1]%hop_length < 1:
        raise ValueError("Less than one step of sliding window movement can be fitted on the time "
            "range of the data. Decrease the hop_length!")
    
    window = np.linspace(0, window_length, window_length, endpoint=False, dtype=int)
    n_times = dataset_train.shape[-1]
    n_steps = int(n_times//hop_length) 

    # create indeces list
    ind_list = [window+num*hop_length for num in range(0, n_steps)]
    #ind_list.append(np.linspace(ind_list[-1][-1]+1, n_times, \
    #    n_times - (ind_list[-1][-1]+1), endpoint=True))
    
    generalizers = []
    shared_train = []
    shared_val = []
    shared_test = []
    backproject_train = []
    backproject_val = []
    backproject_test = []
    
    for inds in ind_list:
        # create featurematrices from windowed datasets
        featuremat_train = dataset2feature_matrix(dataset_train[:,:,:,inds])
        featuremat_val = dataset2feature_matrix(dataset_val[:,:,:,inds])
        featuremat_test = dataset2feature_matrix(dataset_test[:,:,:,inds])
        
        gen_it = copy.deepcopy(generalizer)
        generalizers.append(gen_it.fit(featuremat_train))
        
        shared_train.append(gen_it.project(featuremat_train))
        shared_val.append(gen_it.project(featuremat_val))
        shared_test.append(gen_it.project(featuremat_test))

        backproject_train.append(gen_it.backproject(shared_train[-1]))
        backproject_val.append(gen_it.backproject(shared_val[-1]))
        backproject_test.append(gen_it.backproject(shared_test[-1]))
    
    metadata = {'hop_length': hop_length, 'window_length': window_length,
        'index_list': ind_list}
    projected_data = [shared_train, shared_val, shared_test]
    backprojected_data = [backproject_train, backproject_val, backproject_test]
    
    return projected_data, backprojected_data, generalizers, metadata


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run intersubject generalization from multiviewica '
    'package in window sliding over time. Projecton matrices are learned on time window of train data '
    'and then used to project and backproject same time window of val and test data.')
    parser.add_argument('-inp', '--input_dir', type=str, help='Directory of DATASET MATRICES.')
    parser.add_argument('-out','--output_dir', type=str, help='Directory to store trained geenralizers '
    'projected data, backprojected data and metadata.' )
    parser.add_argument('-method','--method', type=str, help='Which method to use for intersubject '
    'generalization (multiviewica, permica).', default='multiviewica')
    parser.add_argument('-dim_reduction', type=str, help='Method to reduce dimensionality of feature '
    'matrix before applying intersubjct generalization method ("pca" or "srm"). Default = pca', default='pca')
    parser.add_argument('-n_comp','--n_components', type=str, help='Number of components for '
    'dimensionality reduction.')
    parser.add_argument('-wind_len','--window_length',type=int, help='Length of sliding time window. If '
    'None, all the time points are used in one window. Default=None.')
    parser.add_argument('-hop_len','--hop_length', type=int, help='Hop length ==step size of sliding '
    'time window. If none, and window_len is not None, hop_len==wind_len. Default=None.')
    args = parser.parse_args()
     
    # load tarin test and val datasets
    dataset_train = joblib.load(Path(args.input_dir).joinpath('dataset_train.pkl')) 
    dataset_val =  joblib.load(Path(args.input_dir).joinpath('dataset_val.pkl')) 
    dataset_test = joblib.load(Path(args.input_dir).joinpath('dataset_test.pkl')) 
    
    # init intersubject generalizer class with user difined method
    mvica_kwargs = {'tol':1e-5, 'max_iter': 10000}
    if args.n_components=='None':
        args.n_components = None
    else:
        args.n_components = int(args.n_components)
    generalizer = intersubject_generalizer(args.method, args.n_components, \
        args.dim_reduction, mvica_kwargs)

    # fit intersubject generalizer on train data, i.e. learn P and W matrices
    if args.window_length == None:
        args.window_length = dataset_train[-1]
    
    projected_data, backprojected_data, generalizers, metadata=sliding_window(\
        generalizer, dataset_train, dataset_val, dataset_test, args.window_length, args.hop_length)
    
    # save data
    output_dir = Path(args.output_dir) 
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)
    
    names=['_train.pkl', '_val.pkl', '_test.pkl']
    for sh_mat, name in zip(projected_data, names):
        joblib.dump(sh_mat,output_dir.joinpath(('shared'+name))) 
    for bpj_mat, name in zip(backprojected_data, names):
        joblib.dump(bpj_mat, output_dir.joinpath(('backprojected'+name)))
    joblib.dump(generalizers, output_dir.joinpath('generalizers.pkl'))
    joblib.dump(metadata, output_dir.joinpath('metadata.pkl'))
