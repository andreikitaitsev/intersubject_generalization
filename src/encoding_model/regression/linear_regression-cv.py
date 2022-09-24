#! /bin/env/python3
'''
Linear regression with cross-validaiton.
'''

import numpy as np
from pathlib import Path
import joblib
import copy
from load_data import load_dnn_data, load_intersubject_eeg
from sklearn.linear_model import LinearRegression   
from sklearn.preprocessing import StandardScaler
import argparse
import sys

def get_data_snippet(X_tr, Y_tr, ratio, seed):
    '''
    Randomly shuffle X_tr and Y_tr along the image dim 
    and extract the ratio of the data along image dim.
    Inputs:
        X_tr - 2d np array of shape (ims, feat)
        Y_tr - 3d np array of shape (subj, ims, feat)
    Outputs:
        X_tr_chopped - 2d np array of shape (ims*ratio, feat)
        Y_tr_chopped - 2d np array of shape (subj, ims*ratio, feat)
    '''
    n_tr_ims=X_tr.shape[0]
    np.random.seed(seed)
    inds= np.linspace(0, n_tr_ims, n_tr_ims, endpoint=False, dtype=int)
    inds=np.random.permutation(inds)
    X_tr = X_tr[inds, :]
    Y_tr = Y_tr[:, inds, :]

    last_ind= int((Y_tr.shape[1]/100)*ratio)
    X_tr_chopped = X_tr[:last_ind, :]
    Y_tr_chopped = Y_tr[:, :last_ind, :]
    return X_tr_chopped, Y_tr_chopped

def linear_regression_cv(X_tr, X_te, Y_tr, Y_te):
    n_subj=Y_tr.shape[0]
    idxs = np.linspace(0, n_subj, n_subj, endpoint=False, dtype=int)
    Y_te_pred = []
    for te_idx in idxs:
        tr_idx = np.setxor1d(te_idx, idxs)
        regr = LinearRegression()
        regr.fit(X_tr, np.mean(Y_tr[tr_idx,:,:], axis=0))
        Y_te_pred.append(regr.predict(X_te))
    return Y_te_pred

def linear_regression_control(X_tr, X_te, Y_tr, Y_te):
    Y_te_pred = []
    n_subj=Y_tr.shape[0]
    for subj in range(n_subj):
        regr=LinearRegression()
        regr.fit(X_tr, Y_tr[subj,:,:])
        Y_te_pred.append(regr.predict(X_te))
    return Y_te_pred

def linear_regression(X, Y, is_raw, cv=True, ratio=100, seed=0):
    '''Linear regression with leave-one-out cross-validation
    Inputs:
        X - tuple of (x_train, x_test) 
        Y - tuple of (y_train, y_test) 
        is_raw - bool, whether the real data is raw EEG.  
        cv - bool, whether to use CV (not used in control). Default=True.
        ratio - int, ratio of the training data to be left '
        seed - random seed for shuffling the data along the image dim. 
            Default=0.
    Outputs:
        Y_test_pred - predicted responses for the test set.
    '''
    # Unpack X and Y 
    X_tr = X[0]
    X_te = X[1]
    Y_tr = Y[0]
    Y_te = Y[1]
    
    # X.shape = (ims, features)
    # Y.shape: (subj, features, ims) to (subj, ims, features)
    Y_tr = np.transpose(Y_tr, (0, 2, 1))
    Y_te = np.transpose(Y_te, (0, 2, 1))

    # shuffle the featurematrices along the image dim
    if is_raw:
       X_tr, Y_tr = get_data_snippet(X_tr, Y_tr, ratio, seed)

    # Train regression on the average of 9 subjects and predict 10th subject data; 
    # do it in CV loop
    if cv:
        Y_te_pred = linear_regression_cv(X_tr, X_te, Y_tr, Y_te)
    elif not cv:
        Y_te_pred = linear_regression_control(X_tr, X_te, Y_tr, Y_te)
    return Y_te_pred



if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Train linear regression on train set shared space EEG'
    'and predict shared space of the test set individually for every subject.')
    parser.add_argument('-eeg_dir', type=str, help='Directory with eeg in shared space',\
        default="/scratch/akitaitsev/intersubject_generalizeation/linear/multiviewica/")
    parser.add_argument('-dnn_dir', type=str, help="Directory with DNN activations compliant with Ale's"
    " folder structure", default= '/scratch/akitaitsev/encoding_Ale/')
    parser.add_argument('-out_dir', type=str, help='Output directotry to save predicted shared space EEG')
    parser.add_argument('-cv', type=int, default=0, help='Bool, whether to use cross-validation.Default = 0 (False)')
    parser.add_argument('-is_raw', action='store_true', default=False, help='Flag, use raw EEG data.')
    parser.add_argument('-ratio', type=int, default=100, help='Int, ratio of the training data to be left '
    'if is_raw is True, in percents (for the control - encoding model on raw EEG data). Default=100.')
    parser.add_argument('-seed', type=int, default=0, help='Random seed for shuffling the data if is_raw is True. '
    'Default=0.')
    args = parser.parse_args()

    # Load shared space eeg data of shape (subj, features, images)
    if bool(args.is_raw):
        filenames=['featurematrix_train.pkl', 'featurematrix_test.pkl'] 
    elif not bool(args.is_raw):
        filenames = ['shared_train.pkl', 'shared_test.pkl']
    Y_train, Y_test = load_intersubject_eeg(args.eeg_dir, filenames)

    # Load dnn activations
    X_train, X_val, X_test = load_dnn_data('CORnet-S', 1000, args.dnn_dir, skip_val=True)
    
    # linear regression 
    X = [X_train, X_test]
    Y = [Y_train, Y_test]

    Y_test_pred = linear_regression(X, Y, args.is_raw, bool(args.cv), args.ratio, args.seed)

    # save predicted eeg
    path=Path(args.out_dir)
    if not path.is_dir():
        path.mkdir(parents=True)
    joblib.dump(Y_test_pred, (path/('Y_test_predicted_'+'subjectwise'+'.pkl')))    
