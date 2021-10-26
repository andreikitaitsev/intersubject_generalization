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
        Y_te_pred.append(regr.predict(Y_te[te,:,:]))
        if not skip_val:
            Y_val_pred.append(regr.predict(Y_te[te,:,:]))
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
    Y_val_pred, Y_test_pred = linear_regression_cv(X, Y, args.skip_val)

    # save predicted eeg
    none2str = lambda x: str('') if x == None else x
    path=Path(args.out_dir)
    if not path.is_dir():
        path.mkdir(parents=True)
    joblib.dump(Y_test_pred, (path/('Y_test_predicted_'+none2str(args.regr_type)+'.pkl')))    
    joblib.dump(Y_val_pred, (path/('Y_val_predicted_'+none2str(args.regr_type)+'.pkl')))
