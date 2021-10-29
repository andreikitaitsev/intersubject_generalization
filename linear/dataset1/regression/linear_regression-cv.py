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
from sklearn.model_selection import LeaveOneOut

def linear_regression_cv(X, Y, skip_val):
    '''Linear regression with leave-one-out cross-validation
    Inputs:
        X - tuple of (x_train, x_val, x_test) or (x_train, x_test)
        Y - tuple of (y_train, y_val, y_test) or (y_train, y_test)
        skip_val - bool, whether to skip validation set. Default=False.
    Outputs:
        Y_val_pred, Y_test_pred - predicted responses for val and test set.
        If skip_val Y_val_pred is None.
    '''
    # Unpack X and Y 
    if len(X) == 2 and not skip_val:
        raise ValueError('skip_val is False, but the regression got only contains train and test data.') 
    if skip_val:
        X_tr = X[0]
        X_te = X[1]
        Y_tr = Y[0]
        Y_te = Y[1]
    elif not skip_val:
        X_tr = X[0]
        X_val = X[1]
        X_te = X[2]
        Y_tr = Y[0]
        Y_val = Y[1]
        Y_te = Y[2]
    
    # Regression with leave one out CV
    cv_mean = lambda x, ind: np.mean(x[ind,:,:], axis=0)
    Y_te_pred = []
    Y_val_pred = []
    CV = LeaveOneOut()
    for tr, te in CV.split(X_tr):
        regr = LinearRegression()
        regr.fit(np.mean(cv_mean(X_tr, tr), cv_mean(Y_tr, tr))
        Y_te_pred.append(regr.predict(X_te[te,:,:]))
        if not skip_val:
            Y_val_pred.append(regr.predict(X_val[te,:,:]))
        elif skip_val:
            Y_val_pred=None
    return tuple(Y_val_pred), tuple(Y_te_pred)


def linear_regression_cv_raw(X, Y, skip_val, ratio, seed=0):
    '''Linear regression with leave-one-out cross-validation
    Inputs:
        X - tuple of (x_train, x_val, x_test) or (x_train, x_test)
        Y - tuple of (y_train, y_val, y_test) or (y_train, y_test)
        skip_val - bool, whether to skip validation set. Default=False.
        ratio - int, ratio of the tarining images to be left in regression,%
        seed - int - random seed for shuffling the datat aong the image dim.
            Default=0.
    Outputs:
        Y_val_pred, Y_test_pred - predicted responses for val and test set.
        If skip_val Y_val_pred is None.
    '''
    # Unpack X and Y 
    if len(X) == 2 and not skip_val:
        raise ValueError('skip_val is False, but the regression got only contains train and test data.') 
    if skip_val:
        X_tr = X[0]
        X_te = X[1]
        Y_tr = Y[0]
        Y_te = Y[1]
    elif not skip_val:
        X_tr = X[0]
        X_val = X[1]
        X_te = X[2]
        Y_tr = Y[0]
        Y_val = Y[1]
        Y_te = Y[2]
    
    # shuffle the featurematrices along the image dim
    np.random.seed(seed)
    inds= np.linspace(0, Y_tr.shape[-1], Y_tr.shape[-1], endpoint=False, dtype=int)
    inds=np.random.permutation(inds)
    X_tr = X_tr[:,:,inds]
    Y_tr = Y_tr[:,:,inds]
    if not skip_val:
        X_val=X_val[:,:,inds]
        Y_val=Y_val[:,:,inds]
    last_ind= int((Y_tr.shape[-1]/100)*ratio)
    X_tr = X_tr[:,:,:last_ind]
    Y_tr = Y_tr[:,:,:last_ind]
    if not skip_val:
        X_val = X_val[:,:,:last_ind]
        Y_val = Y_val[:,:,:last_ind]

    # Regression with leave one out CV
    cv_mean = lambda x, ind: np.mean(x[ind,:,:], axis=0)
    Y_te_pred = []
    Y_val_pred = []
    CV = LeaveOneOut()
    for tr, te in CV.split(X_tr):
        regr = LinearRegression()
        regr.fit(np.mean(cv_mean(X_tr, tr), cv_mean(Y_tr, tr))
        Y_te_pred.append(regr.predict(X_te[te,:,:]))
        if not skip_val:
            Y_val_pred.append(regr.predict(X_val[te,:,:]))
        elif skip_val:
            Y_val_pred=None
    return tuple(Y_val_pred), tuple(Y_te_pred)


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Train linear regression on train set shared space EEG'
    'and predict shared space of the test set individually for every subject.')
    parser.add_argument('-eeg_dir', type=str, help='Directory with eeg in shared space',\
        default="/scratch/akitaitsev/intersubject_generalizeation/linear/multiviewica/")
    parser.add_argument('-dnn_dir', type=str, help="Directory with DNN activations compliant with Ale's"
    " folder structure", default= '/scratch/akitaitsev/encoding_Ale/')
    parser.add_argument('-out_dir', type=str, help='Output directotry to save predicted shared space EEG')
    parser.add_argument('-skip_val', action='store_true', default=False, help='Flag, whether to skip val set.')
    parser.add_argument('-is_raw', action='store_true', default=False, help='Flag, use raw EEG data.')
    parser.add_argument('-ratio', type=int, default=100, help='Int, ratio of the training data to be left '
    'if is_raw is True, in percents (for the control - encoding model on raw EEG data). Default=100.')
    parser.add_argument('-seed', type=int, default=0, help='Random seed for shuffling the data if is_raw is True. '
    'Default=0.')
    args = parser.parse_args()

    # Load shared space eeg data of shape (subj, features, images)
    if bool(args.is_raw):
        filenames=['featurematrix_train.pkl', 'featurematrix_val.pkl','featurematrix_test.pkl'] 
    elif not bool(args.is_raw):
        filenames = ['shared_train.pkl', 'shared_val.pkl', 'shared_test.pkl']
        
    Y_train, Y_val, Y_test = load_intersubject_eeg(args.eeg_dir, filenames, skip_val)

    # Load dnn activations
    if args.skip_val:
        X_train, X_test = load_dnn_data('CORnet-S', 1000, args.dnn_dir)
        X_val=None
    elif not args.skip_val:
        X_train, X_val, X_test = load_dnn_data('CORnet-S', 1000, args.dnn_dir)
    
    # linear regression with leave-one-out CV
    if args.skip_val:
        X = [X_train, X_test]
        Y = [Y_train, Y_test]
    elif not args.skip_val
        X = [X_train, X_val, X_test]
        Y = [Y_train, Y_val, Y_test]


    if args.is_raw:
        Y_val_pred, Y_test_pred = linear_regression_cv_raw(X, Y, args.skip_val, args.ratio, args.seed)
    elif not args.is_raw:
        Y_val_pred, Y_test_pred = linear_regression_cv(X, Y, args.skip_val)

    # save predicted eeg
    none2str = lambda x: str('') if x == None else x
    path=Path(args.out_dir)
    if not path.is_dir():
        path.mkdir(parents=True)
    joblib.dump(Y_test_pred, (path/('Y_test_predicted_'+none2str(args.regr_type)+'.pkl')))    
    joblib.dump(Y_val_pred, (path/('Y_val_predicted_'+none2str(args.regr_type)+'.pkl')))
