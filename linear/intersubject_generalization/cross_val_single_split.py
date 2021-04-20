#! /bin/env/python3
'''Run intersubject generalization with cross validation for single split.
Parallelize using SLURM array.'''

import numpy as np
import joblib
import copy
import warnings
from pathlib import Path
from linear_intersubject_generalization_utils import intersubject_generalizer
from sklearn.model_selection import KFold
import sys
sys.path.append('/home/akitaitsev/code/intersubject_generalization/linear/create_featurematrices')
from create_featurematrix import dataset2feature_matrix
from linear_intersubject_generalization_utils import intersubject_generalizer

__all__=['create_splits', 'cross_val_single_split']

def create_splits(featuremat_train, featuremat_val, featuremat_test, n_splits):
    '''Split the data into n_splits.
    Inputs:
        featuremat_train, featuremat_val, featuremat_test - 3d numpy arrays of shape
        (subjs, features, images)
        n_splits - int, n splits for cross validation over subjects
    Outputs:
        splits_train, splits_val, splits_test - lists of len n_splits
    '''
    splits_train = []
    splits_val=[]
    splits_test = []

    kfold=KFold(n_splits=n_splits)
    n_subjects = featuremat_train.shape[0]
    dummy_ar = np.linspace(0, n_subjects, n_subjects, endpoint=False, dtype=int)
    expand = lambda x: np.expand_dims(x, axis=0) if np.ndim(x)==2 else x 
    for train, test in kfold.split(dummy_ar):
        splits_train.append(featuremat_train[train,:,:])
        splits_val.append(featuremat_val[test,:,:])
        splits_test.append(featuremat_test[test,:,:])
    return splits_train, splits_val, splits_test


def cross_val_single_split(generalizer, splits_train, splits_val, splits_test, split2use):
    '''
    Fit and project data with intersubject generalizer object with k fold CV.
    Inputs:
        generalizer - configured intersubject_generalizer class instance
        splits_train, splits_val, splits_test - lists of len(n_splits) containing 
        3d numpy arrays of featurematrices for every split
        split2use - int, which split to use for intersubject generalization
    Outputs:
        shared - tuple of shared train, val and test shared space data for split2use
        backprojected - tuple of backprojected train, val and test data for split2use
        gener - trained intersubject generalizer for split2use 
        metadata - dictionary with parameters used 
    '''
    
    gener = copy.deepcopy(generalizer)
    gener.fit(splits_train[split2use])
    shared_train = gener.project(splits_train[split2use])
    shared_val = gener.project(splits_val[split2use])
    shared_test = gener.project(splits_test[split2use])
    back_train = gener.backproject(shared_train)
    back_val = gener.backproject(shared_val)
    back_test = gener.backproject(shared_test)
    
    shared = (shared_train, shared_val, shared_test)
    backprojected = (back_train, back_val, back_test)
    metadata = {'split2use': split2use}
    return shared, backprojected, gener, metadata

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run intersubject generalization from multiviewica '
    'package in with cross validation over subjects. Projecton matrices are learned on train data '
    'and then used to project and backproject out of sample train, val and test data.')
    parser.add_argument('-inp', '--input_dir', type=str, help='Directory of Feature Matrices.')
    parser.add_argument('-out','--output_dir', type=str, help='Directory to store trained geenralizers '
    'projected data, backprojected data and metadata.' )
    parser.add_argument('-method','--method', type=str, help='Which method to use for intersubject '
    'generalization (multiviewica, permica).', default='multiviewica')
    parser.add_argument('-dim_reduction', type=str, help='Method to reduce dimensionality of feature '
    'matrix before applying intersubjct generalization method ("pca" or "srm"). Default = pca', default='pca')
    parser.add_argument('-n_comp','--n_components', type=str, help='Number of components for '
    'dimensionality reduction.')
    parser.add_argument('-n_splits', type=int, default=7, help='N splits in cross validation over subjects.')
    parser.add_argument('-split2use', type=int, help='No split to use.')
    args = parser.parse_args()
     
    # load tarin test and val featurematrices
    featuremat_train = joblib.load(Path(args.input_dir).joinpath('featurematrix_train.pkl')) 
    featuremat_val =  joblib.load(Path(args.input_dir).joinpath('featurematrix_val.pkl')) 
    featuremat_test = joblib.load(Path(args.input_dir).joinpath('featurematrix_test.pkl')) 
    
    # init intersubject generalizer class with user difined method
    method_kwargs = {'tol':1e-5, 'max_iter': 10000}
    if args.n_components=='None':
        args.n_components = None
    else:
        args.n_components = int(args.n_components)
    generalizer = intersubject_generalizer(args.method, args.n_components, \
        args.dim_reduction, method_kwargs)
    
    # Divide data into n_splits
    splits_train, splits_val, splits_test = create_splits(featuremat_train, featuremat_val, \
        featuremat_test, args.n_splits)
    
    # Run intersubject generalization for split2use  
    shared, backprojected, gener, metadata = cross_val_single_split(generalizer, \
        splits_train, splits_val, splits_test, args.split2use) 
    
    # save data for split2use
    output_dir = Path(args.output_dir) 
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)
    
    names=['_train_', '_val_', '_test_']
    for sh_mat, name in zip(shared, names):
        joblib.dump(sh_mat,output_dir.joinpath(('shared'+name + str(args.split2use)+'.pkl'))) 
    for bpj_mat, name in zip(backprojected, names):
        joblib.dump(bpj_mat, output_dir.joinpath(('backprojected'+name+str(args.split2use)+'.pkl')))
    joblib.dump(gener, output_dir.joinpath(('generalizer'+str(args.split2use)+'.pkl')))
    joblib.dump(metadata, output_dir.joinpath(('metadata_'+str(args.split2use)+'.pkl')))
