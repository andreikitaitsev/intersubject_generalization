#! /bin/env/python3
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
__all__=['cross_val']


def generalize_cross_val(generalizer, featuremat_train, featuremat_val, featuremat_test, n_splits):
    '''
    Fit and project data with intersubject generalizer object with k fold CV.
    Inputs:
        generalizer - configured intersubject_generalizer class instance
        featuremat_train 
        featuremat_val
        featuremat_test -
                         3d numpy arrays of featurematrices of shapes (subj, n_features, n_images)
        n_splits = int, n splits across subejct dimension to be used in CV
    Outputs:
        shared - tuple of shared train, val and test shared space data
        backprojected - tuple of backprojected train, val and test data
        geners - tuple of trained intersubject generalizers for each CV fold
        metadata - dictionary with parameters used 
    '''
    
    geners = []
    shared_train = []
    shared_val = []
    shared_test = []
    back_train=[]
    back_val = []
    back_test = []

    kfold=KFold(n_splits=n_splits)
    n_subjects = featuremat_train.shape[0]
    dummy_ar = np.linspace(0, n_subjects, n_subjects, endpoint=False, dtype=int)
    expand = lambda x: np.expand_dims(x, axis=0) if np.ndim(x)==2 else x 
    for train, test in kfold.split(dummy_ar):
        geners.append(copy.deepcopy(generalizer))
        geners[-1].fit(featuremat_train[train,:,:])
        
        shared_train.append(geners[-1].project(expand(featuremat_train[train,:,:])))
        shared_val.append(geners[-1].project(expand(featuremat_val[test,:,:])))
        shared_test.append(geners[-1].project(expand(featuremat_test[test,:,:])))
        
        back_train.append(geners[-1].backproject(shared_train[-1]))
        back_val.append(geners[-1].backproject(shared_val[-1]))
        back_test.append(geners[-1].backproject(shared_test[-1]))
    shared = (shared_train, shared_val, shared_test)
    backprojected= (back_train, back_val, back_test)
    metadata = {'n_splits':n_splits}
    return shared, backprojected, tuple(geners), metadata


    
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
    
    # Run intersubject generalization   
    projected_data, backprojected_data, generalizers, metadata= generalize_cross_val(\
        generalizer, featuremat_train, featuremat_val, featuremat_test,args.n_splits) 
    
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
