#! /bin/env/python3
'''Script to train linear regression on train data in shared 
space and predict test data subjectwise in shared space
on the data AVERAGED across subjects.
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
from sklearn.model_selection import KFold


def linear_regression_simple(X_tr, X_test, Y_train, Y_test, regr_type):
    '''Perform linear regression on the whole featurespace.'''
    # transpose eeg to shape (subj,images, features)
    Y_train = np.transpose(np.array(Y_train), (0,2,1))
    Y_test = np.transpose(np.array(Y_test), (0,2,1))
    scaler = StandardScaler()
    regr = LinearRegression()
    trained_regrs = []
    if regr_type == 'average':
        Y_train = np.mean(Y_train, axis=0) 
        Y_test = np.mean(Y_test, axis=0) 
        # fit regr
        regr.fit(scaler.fit_transform(X_tr), Y_train) 
        Y_test_pred= regr.predict(scaler.fit_transform(X_test))
        trained_regrs.append(regr)
    elif regr_type == 'subjectwise':
        Y_test_pred=[]
        for subj in range(7):
            trained_regrs.append(copy.deepcopy(regr))
            trained_regrs[-1].fit(scaler.fit_transform(X_tr), Y_train[subj])
            Y_test_pred.append(trained_regrs[-1].predict(scaler.fit_transform(X_test)))
        Y_test_pred = np.stack(Y_test_pred, axis=0)
    return Y_test_pred, trained_regrs


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Train linear regression on train set shared space EEG'
    'and predict shared space of the test set individually for every subject.')
    parser.add_argument('-eeg_dir', type=str, help='Directory with eeg in shared space',\
        default="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset2/multiviewica/")
    parser.add_argument('-dnn_dir', type=str, help="Directory with DNN activations compliant with Ale's"
    " folder structure", default= '/scratch/akitaitsev/encoding_Ale/dataset2/')
    parser.add_argument('-out_dir', type=str, help='Output directotry to save predicted shared space EEG')
    parser.add_argument('-is_raw', action='store_true', default=False, help='Bool, raw data flag.')
    parser.add_argument('-regr_type', type=str, default=None, help='average/subjectwise/cv. Whether to perform regression on '
    'data averaged across all subjects, subjectwiese or in cross-validation framework.')
    args=parser.parse_args()

    # Load shared space eeg data of shape (subj, features, images)
    if args.is_raw:
        filenames=['featurematrix_train.pkl', 'featurematrix_test.pkl'] 
    elif not args.is_raw:
        filenames = ['shared_train.pkl', 'shared_test.pkl']
    try:
        Y_train, Y_test = load_intersubject_eeg(args.eeg_dir, filenames)
    except FileNotFoundError:
        print('No such method: '+ str(args.eeg_dir))
        sys.exit(0)    
    
    # Load dnn activations
    X_tr, X_test = load_dnn_data('CORnet-S', 1000, args.dnn_dir)

    # linear regression    
    Y_test_pred, trained_regrs = linear_regression_simple(X_tr, X_test, Y_train, Y_test,\
        args.regr_type)

    # save predicted eeg
    none2str = lambda x: str('') if x == None else x
    path=Path(args.out_dir)
    if not path.is_dir():
        path.mkdir(parents=True)
    joblib.dump(Y_test_pred, (path/('Y_test_predicted_'+none2str(args.regr_type)+'.pkl')))    
    joblib.dump(trained_regrs, (path/('trained_regression_'+none2str(args.regr_type)+'.pkl')))
