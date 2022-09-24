#! /bin/bash
'''
Script provides wrapper around multiviewica functions to use with sklearn-like 
syntax (.fit, .project, etc) convinient for train-test setup.
If called as main, allows to run multiviewica or permica on featurematrix of
shape (subj, features, samples). Use create_featurematrix.py to create feature matrix.
Note, that script assumes multiviewica installed!
'''
from pathlib import Path
import numpy as np
import joblib
try:
    import multiviewica as mvica
except:
    print('Multiviewica is not installed. See  \n'
    'https://github.com/hugorichard/multiviewica/blob/master/README.md \n'
    'for installation instructions.')

class intersubject_generalizer():    
    def __init__(self, method, n_components, dimension_reduction, method_kwargs=None):
        '''Create an object for intersubject generalization.
        Attributes:
            method - str, multiviewica, permica or groupica,
                     method to use for intersubject generalization
            n_components - int, number of components to use in 
                           dimensionality reduction
            dimension_reduction - str, 'pca' or 'srm' - how to reduce dimensions
            method_kwargs - dict of additional kwargs for the intersubject 
                            generalization method
            P - dimensionality reduction matrix for every subject(pca or srm,
                see multiview ica funcs
            W - unmixing matrices for every subject
            S - estimated average over all subjects shared source of shape 
               (shared_features, images)
            PW - composite dimensionality reduction and unmixing matrices for
                 every subject.

        User guide:
        1. calll .fit method on the dataset you want to estimate dim reduction and 
        unmixing matrices on (train set). 
        2. call .project on the test dataset to use learned dim reduction and 
        unmixing matrices to project in into the shared space.
        3. Optional, do the analysis in shared space.
        4. call .backproject on the dataset in shared space to project it back
        to original (subject-specific) space.
        Use self.P, self.W, self.S to access the outputs of multiviewica module.
        '''
        self.method = eval('mvica.'+method)
        self.n_components = n_components
        self.dimension_reduction = dimension_reduction
        self.method_kwargs = method_kwargs 
        self.P = None
        self.W = None
        self.S = None
        self.WP = None
  
    def fit(self, dataset):
        '''Learn unmixing matrices
           Input:
                dataset - 3n numpy array of shape (subj, features, images)
        '''
        if not self.method_kwargs == None:
            self.P, self.W, self.S = self.method(dataset, n_components=self.n_components,\
                dimension_reduction=self.dimension_reduction, 
                **self.method_kwargs)
        else:
            self.P, self.W, self.S = self.method(dataset, n_components=self.n_components,\
                dimension_reduction=self.dimension_reduction) 
        self.WP = np.stack([self.W[i].dot(self.P[i]) for i in range(dataset.shape[0])], axis=0)
    
    def project(self, data):
        '''Project data into a shared space using unmixing matrices learned during .fit
        Inputs:
            data - 3d numpy array (subj, features, images) of data in origianl space
        Outputs:
            projected_data - 3d numpy array (subj, shared_features, images) of data in
                             shared space.
        To get estimated shared space take the mean of the output over 1st dim (subjects).
        '''
        shared_space_data = []
        for i in range(data.shape[0]):
            shared_space_data.append(np.dot(self.WP[i], data[i]))
        return np.stack(shared_space_data, axis=0)
    
    def backproject(self, data):
        '''
        Backprojects data from shared space into individual space by multiplying it 
        with the pseudoinverse of the unmixing matrices.
        Inputs:
            data - 3d numpy array (subj, shared_features, images) of data in shared 
                   space
        Outputs:
            backprojected_data - 3d numpy array (subj, features, images) of data in
                                 subject-specific (original) space
        '''
        backprojected_data = []
        for i in range(data.shape[0]):
            backprojected_data.append(np.dot(np.linalg.pinv(self.WP[i]), data[i]))
        backprojected_data = np.stack(backprojected_data,axis=0)
        return backprojected_data


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run intersubject generalization from multiviewica '
    'package. Projecton matrices are learned on train data and then used to project and backproject '
    'val and test data.')
    parser.add_argument('-inp', '--input_dir', type=str, help='Directory of feature matrices. Time '
    'window is scanned from directory path automatically.')
    parser.add_argument('-out','--output_dir', type=str, help='Directory to store projected data, '
    'backprojected data and trained generalized object')
    parser.add_argument('-method','--method', type=str, help='Which method to use for intersubject '
    'generalization (multiviewica, permica). Groupica is not supported yet.', default='multiviewica')
    parser.add_argument('-dim_reduction', type=str, help='Method to reduce dimensionality of feature '
    'matrix before applying intersubjct generalization method ("pca" or "srm"). Default = pca', default='pca')
    parser.add_argument('-n_comp','--n_components', type=str, help='Number of components for '
    'dimensionality reduction. Default = 200', default=200)
    args = parser.parse_args()
     
    # load tarin test and val feature matrices
    featuremat_train = joblib.load(Path(args.input_dir).joinpath('featurematrix_train.pkl')) 
    featuremat_val =  joblib.load(Path(args.input_dir).joinpath('featurematrix_val.pkl')) 
    featuremat_test = joblib.load(Path(args.input_dir).joinpath('featurematrix_test.pkl')) 
    
    # init intersubject generalizer class with user difined method
    mvica_kwargs = {'tol':1e-5, 'max_iter': 10000}
    if args.n_components=='None':
        args.n_components = None
    else:
        args.n_components = int(args.n_components)
    generalizer = intersubject_generalizer(args.method, args.n_components, \
        args.dim_reduction, mvica_kwargs)

    # fit intersubject generalizer on train data, i.e. learn P and W matrices
    generalizer.fit(featuremat_train)
    
    shared_train = generalizer.project(featuremat_train)
    shared_test = generalizer.project(featuremat_test)
    shared_val = generalizer.project(featuremat_val)
    
    backprojected_train = generalizer.backproject(shared_train)
    backprojected_test = generalizer.backproject(shared_test)
    backprojected_val =  generalizer.backproject(shared_val)
    
    # save data
    output_dir = Path(args.output_dir) 
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)
    joblib.dump(shared_train, output_dir.joinpath('shared_train.pkl'))
    joblib.dump(shared_test, output_dir.joinpath('shared_test.pkl'))
    joblib.dump(shared_val, output_dir.joinpath('shared_val.pkl'))

    joblib.dump(backprojected_train,output_dir.joinpath('backprojected_train.pkl'))
    joblib.dump(backprojected_test, output_dir.joinpath('backprojected_test.pkl'))
    joblib.dump(backprojected_val, output_dir.joinpath('backprojected_val.pkl'))
    
    joblib.dump(generalizer, output_dir.joinpath('trained_generalizer.pkl'))
