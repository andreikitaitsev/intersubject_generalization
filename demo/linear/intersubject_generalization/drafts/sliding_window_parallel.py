#! /bin/env/python3
'''
Script implements intersubject generalization with sliding window over time points.
For each if the time point ranges in the raw data the featurematrix (channels x time) 
is created and intersubject generalizer object is fitted on each of these snippets.
'''
import numpy as np
import joblib 
import copy
from pathos.multiprocessing import ProcessingPool as Pool
#from multiprocessing.pool import ThreadPool as Pool
#import multiprocessing as mp
import warnings
from pathlib import Path
# directory with feature matrices scripts
import sys
sys.path.append('/home/akitaitsev/code/intersubject_generalization/linear/create_featurematrices')
from create_featurematrix import dataset2feature_matrix
from linear_intersubject_generalization_utils import intersubject_generalizer

__all__ =  ['create_data_snippets','process_data_snippet','process_snippets_parallel', \
    'sliding_window_parallel'] 


def create_data_snippets(dataset_train, dataset_val, dataset_test, window_length, hop_length):
    '''
    Chop train, val and test set data into snippets for parallel computing based on
    time window and hop lengths. Each snippet is data of within 1 time window.
    Inputs:
        dataset_train - 4d numpy array of shape (subj, images, channels, times)
        dataset_val - 4d numpy array of shape (subj, images, channels, times)
        dataset_test - 4d numpy array of shape (subj, images, channels, times)
        window_length - int, length of liding window in samples
        hop_length - int, step size of sliding window replacement. If None,
                     hop_length==window_length. Default=None
    Outputs:
        snippets_train 
        snippets_val
        snippets_test
                    - lists of featurematrices of shape (subjs, features, images) where features
                    are created by calling dataset2feature_matrix on every time windowed data snippet 
                    for train, val and test data.
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
    
    snippets_train = []
    snippets_val = []
    snippets_test = []

    for inds in ind_list:
        # create featurematrices from windowed datasets
        snippets_train.append(dataset2feature_matrix(dataset_train[:,:,:,inds]))
        snippets_val.append(dataset2feature_matrix(dataset_val[:,:,:,inds]))
        snippets_test.append(dataset2feature_matrix(dataset_test[:,:,:,inds]))

    metadata = {'hop_length': hop_length, 'window_length': window_length,
        'index_list': ind_list}
    return snippets_train, snippets_val, snippets_test, metadata


def process_data_snippet(generalizer, snippet_train, snippet_val, snippet_test):
    '''Apply intersubject generalization at single data snippet. Creates deepcopy of
    intersubject generalizer object and fits it on train snippet and projects and 
    backprojects it on val and test data.
    Inputs:
        generalizer - intersubject generalizer object to be fitted on tarin data
        snippet_train, snippet_val, snippet_test - 
            3d numpy arrays of shape 
            (subj,features, images) for train, val and test data. Outputs of 
            create_data_snippets function.
    Outputs:
        data - dictionary with fields(projected_train, projected_val, projected_test
               backprojected_train, backprojected_val, backprojected_test, generalizer)
    '''
    generalizer_=copy.deepcopy(generalizer)
    generalizer_.fit(snippet_train)
    projected_train=generalizer_.project(snippet_train)
    projected_val=generalizer_.project(snippet_val)
    projected_test=generalizer_.project(snippet_test)
    backprojected_train=generalizer_.backproject(projected_train)
    backprojected_val=generalizer_.backproject(projected_val)
    backprojected_test=generalizer_.backproject(projected_test)
    data = {'projected_train': projected_train, 'projected_val': projected_val,\
    'projected_test':projected_test, 'backprojected_train': backprojected_train, \
        'backprojected_val':backprojected_val, 'backprojected_test': backprojected_test,\
        'generalizer': generalizer_}
    return data

def process_snippets_parallel(generalizer, metadata, snippets_train, snippets_val, snippets_test): 
    '''
    '''
    pool=Pool(len(metadata["index_list"])) # as many processes as there are windows
    results=[pool.pipe(process_data_snippet,*(generalizer, s_tr, s_val, s_test)) \
        for s_tr, s_val, s_test in zip(snippets_train, snippets_val, snippets_test)]
    pool.close()
    return results


def sliding_window_parallel(generalizer, dataset_train, dataset_val, dataset_test,\
    window_length, hop_length=None):
    '''Run sliding window intersubject generalization parallely on each time window.
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
    
    # chop input data into snippets (one snippet=one time window)
    snippets_train, snippets_val, snippets_test, metadata = create_data_snippets(dataset_train,\
        dataset_val, dataset_test, window_length, hop_length)
    
    # run intersubject generalization on each snippet parallely
    # results - list of dictionaries for each time window
    results = process_snippets_parallel(generalizer, metadata, snippets_train, snippets_val, snippets_test)
    
    # concatenate results
    projected_data=[]
    backprojected_data=[]
    generalizers=[]

    for window in results:
        projected_data.append([window["projected_train"], window["projected_val"], \
            window["projected_test"]])
        backprojected_data.append([window["backprojected_train"], window["backprojected_val"],\
            window["backprojected_test"]])
        generalizers.append(window["generalizer"])
    return projected_data, backprojected_data, generalizers, metadata


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run sliding time window intersubject generalization '
    'from multiviewica package in parallel for each time window. Projecton matrices are learned on time'
    'window of train data and then used to project and backproject same time window of val and test data.')
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

    if args.window_length == None:
        args.window_length = dataset_train[-1]
    
    projected_data, backprojected_data, generalizers, metadata=sliding_window_parallel(\
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
