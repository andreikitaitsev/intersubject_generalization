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

__all__ = ['linear_regression']

def _linear_regression_simple(X_tr, X_val, X_test, Y_train, Y_val, Y_test, regr_type):
    '''Perform linear regression on the whole featurespace.'''
    # transpose eeg to shape (subj,images, features)
    Y_train = np.transpose(np.array(Y_train), (0,2,1))
    Y_val = np.transpose(np.array(Y_val), (0,2,1))
    Y_test = np.transpose(np.array(Y_test), (0,2,1))

    n_subj=Y_train.shape[0]
    scaler = StandardScaler()
    regr = LinearRegression()
    trained_regrs = []
    if regr_type == 'average':
        Y_train = np.mean(Y_train, axis=0) 
        Y_val = np.mean(Y_val, axis=0) 
        Y_test = np.mean(Y_test, axis=0) 
        # fit regr
        regr.fit(scaler.fit_transform(X_tr), Y_train) 
        Y_val_pred = regr.predict(scaler.fit_transform(X_val))
        Y_test_pred= regr.predict(scaler.fit_transform(X_test))
        trained_regrs.append(regr)
    elif regr_type == 'subjectwise':
        Y_val_pred=[]
        Y_test_pred=[]
        for subj in range(n_subj):
            trained_regrs.append(copy.deepcopy(regr))
            trained_regrs[-1].fit(scaler.fit_transform(X_tr), Y_train[subj])
            Y_val_pred.append(trained_regrs[-1].predict(scaler.fit_transform(X_val)))
            Y_test_pred.append(trained_regrs[-1].predict(scaler.fit_transform(X_test)))
        Y_val_pred = np.stack(Y_val_pred, axis=0)
        Y_test_pred = np.stack(Y_test_pred, axis=0)
    return Y_val_pred, Y_test_pred, trained_regrs



def _linear_regression_sliding_window(X_tr, X_val, X_test, Y_train, Y_val, Y_test, regr_type):
    '''Perform linear regression on sliding time window data.'''
    Y_train_pred=[]
    Y_val_pred=[]
    Y_test_pred=[]
    scalers = []
    trained_regrs=[]
    scaler = StandardScaler()
    regr = LinearRegression()
    for Y_train_it, Y_val_it, Y_test_it in zip(Y_train, Y_val, Y_test):
        # transpose eeg to shape (subj,images, features)
        Y_train_it = np.transpose(np.array(Y_train_it), (0,2,1))
        Y_val_it = np.transpose(np.array(Y_val_it), (0,2,1))
        Y_test_it = np.transpose(np.array(Y_test_it), (0,2,1))
        n_subj=Y_train.shape[0]
        if regr_type == 'average':
            Y_train_it = np.mean(Y_train_it, axis=0) 
            Y_val_it = np.mean(Y_val_it, axis=0) 
            Y_test_it = np.mean(Y_test_it, axis=0) 
            # fit regr
            scalers.append(copy.deepcopy(scaler))
            trained_regrs.append(copy.deepcopy(regr).fit(scalers[-1].\
                fit_transform(X_tr), Y_train_it)) 
            Y_val_pred.append(trained_regrs[-1].predict(scalers[-1].fit_transform(X_val)))
            Y_test_pred.append(trained_regrs[-1].predict(scalers[-1].fit_transform(X_test)))
        elif regr_type == 'subjectwise':
            Y_train_pred_it = []
            Y_val_pred_it = []
            Y_test_pred_it = []
            scalers_it=[]
            regrs_it=[]
            for subj in range(n_subj):
                regrs_it.append(copy.deepcopy(regr))
                scalers_it.append(copy.deepcopy(scaler))
                regrs_it[-1].fit(scalers_it[-1].fit_transform(X_tr), Y_train_it[subj])
                Y_val_pred_it.append(regrs_it[-1].predict(scalers_it[-1].fit_transform(X_val)))
                Y_test_pred_it.append(regrs_it[-1].predict(scalers_it[-1].fit_transform(X_test)))
            Y_val_pred_it = np.stack(Y_val_pred_it, axis=0)
            Y_test_pred_it = np.stack(Y_test_pred_it, axis=0)
            Y_val_pred.append(Y_val_pred_it)
            Y_test_pred.append(Y_test_pred_it)
            scalers.append(scalers_it)
            trained_regrs.append(regrs_it)
    return Y_val_pred, Y_test_pred, trained_regrs


def _linear_regression_cv_regr(X_tr, X_val, X_test, Y_train, Y_val, Y_test, n_splits):
    '''
    Perform linear regression using cross-validaiton over subjects.
    Regression trained on average of n-1 subjects is used to predict out of sample
    nth subject's data.

    The convention is to call regr_type "subejctwise" in this case, though it's not
    the same as subjectwise in another regression functions.
    
    Inputs:
        X_tr, X_val, X_test
        Y_tarin, Y_val, Y_test - lists, outputs of cross_val function
        n_splits - int, n splits to be used in cross-validation over subejcts
    Outputs:
        Y_val_pred 
        Y_test_pred
        trained_regrs
    '''
    Y_train_pred=[]
    Y_val_pred=[]
    Y_test_pred=[]
    scalers = []
    trained_regrs=[]
    scaler = StandardScaler()
    regr = LinearRegression()
    
    # transpose eeg to shape (subj, images, features)
    Y_train = np.transpose(np.array(Y_train), (0,2,1))
    Y_val = np.transpose(np.array(Y_val), (0,2,1))
    Y_test = np.transpose(np.array(Y_test), (0,2,1))
    
    kf = KFold(n_splits=n_splits)
    dummy_ar=np.linspace(0, Y_train.shape[0], Y_train.shape[0], endpoint=False, dtype=int)
    for train, test in kf.split(dummy_ar):
        Y_train_it = np.mean(Y_train[train], axis=0) 
        # fit regr
        scalers.append(copy.deepcopy(scaler))
        trained_regrs.append(copy.deepcopy(regr).fit(scalers[-1].\
            fit_transform(X_tr), Y_train_it)) 
        Y_val_pred.append(trained_regrs[-1].predict(scalers[-1].fit_transform(X_val)))
        Y_test_pred.append(trained_regrs[-1].predict(scalers[-1].fit_transform(X_test)))
    return Y_val_pred, Y_test_pred, trained_regrs

def _linear_regression_cv_gener(X_tr, X_val, X_test, Y_train, Y_val, Y_test):
    '''Perform linear regression on data obtained with intersubject generalization 
    with cross-validation.
    The projection matrices were learned on (n-1) subjects and used to project nth 
    subject into the shared space. 
    The regression is fitted on train data projected into shared space and used to 
    predict out of sample subject data for each cross-validation split.
    '''
    Y_val_pred = []
    Y_test_pred=[]

    scaler = StandardScaler()
    regr = LinearRegression()
    trained_regrs = []
    for split in range(7):
        # transpose eeg to shape (subj,images, features)
        Y_train_it = np.transpose(np.array(Y_train[split]), (0,2,1))
        Y_val_it = np.transpose(np.array(Y_val[split]), (0,2,1))
        Y_test_it = np.transpose(np.array(Y_test[split]), (0,2,1))
        
        trained_regrs.append(copy.deepcopy(regr))
        trained_regrs[-1].fit(scaler.fit_transform(X_tr), np.mean(Y_train_it, axis=0))
        Y_val_pred.append(trained_regrs[-1].predict(scaler.fit_transform(X_val)))
        Y_test_pred.append(trained_regrs[-1].predict(scaler.fit_transform(X_test)))
    return Y_val_pred, Y_test_pred, trained_regrs


def _linear_regression_incr_train_data(X_tr, X_val, X_test, Y_train, Y_val, Y_test, regr_type):
    '''
    Perform linear regression incrementally increased amout of training data images.
    The only difference from _linear_regression_simple is that the regression shall be
    fitted on part of X_tr images corresponding to those used in trainig eeg.
    '''
    # transpose eeg to shape (subj,images, features)
    Y_train = np.transpose(np.array(Y_train), (0,2,1))
    Y_val = np.transpose(np.array(Y_val), (0,2,1))
    Y_test = np.transpose(np.array(Y_test), (0,2,1))

    scaler = StandardScaler()
    regr = LinearRegression()
    trained_regrs = []
    if regr_type == 'average':
        Y_train = np.mean(Y_train, axis=0) 
        Y_val = np.mean(Y_val, axis=0) 
        Y_test = np.mean(Y_test, axis=0) 
        # fit regr
        regr.fit(scaler.fit_transform(X_tr[0:Y_train.shape[0], :]), Y_train) 
        Y_val_pred = regr.predict(scaler.fit_transform(X_val))
        Y_test_pred= regr.predict(scaler.fit_transform(X_test))
        trained_regrs.append(regr)
    elif regr_type == 'subjectwise':
        Y_val_pred=[]
        Y_test_pred=[]
        for subj in range(7):
            trained_regrs.append(copy.deepcopy(regr))
            trained_regrs[-1].fit(scaler.fit_transform(X_tr[0: Y_train[subj].shape[0],:]),\
                Y_train[subj])
            Y_val_pred.append(trained_regrs[-1].predict(scaler.fit_transform(X_val)))
            Y_test_pred.append(trained_regrs[-1].predict(scaler.fit_transform(X_test)))
        Y_val_pred = np.stack(Y_val_pred, axis=0)
        Y_test_pred = np.stack(Y_test_pred, axis=0)
    return Y_val_pred, Y_test_pred, trained_regrs

def linear_regression(X_tr, X_val, X_test, Y_train, Y_val, Y_test, regr_type, \
    sliding_window, n_splits, cv_regr, cv_gener, incr_train_data, learn_pr_incr):
    '''Linear regression wrapper function. Performs either simple or sliding window
    or cross-validation linear regression.
    Inputs:
    Outputs:
    '''
    if not sliding_window and n_splits==None and not cv_gener and not incr_train_data and not\
            learn_pr_incr:
        Y_val_pred, Y_test_pred, trained_regrs = _linear_regression_simple(X_tr, X_val, X_test,\
            Y_train, Y_val, Y_test, regr_type)
    
    elif sliding_window and n_splits == None and not cv_gener and not incr_train_data and not\
            learn_pr_incr:
        Y_val_pred, Y_test_pred, trained_regrs = _linear_regression_sliding_window(X_tr, X_val, \
            X_test, Y_train, Y_val, Y_test, regr_type)
        
    elif cv_regr and not sliding_window and n_splits != None and not incr_train_data and not  \
            learn_pr_incr:
        Y_val_pred, Y_test_pred, trained_regrs = _linear_regression_cv_regr(X_tr, X_val, \
            X_test, Y_train, Y_val, Y_test, n_splits)
    
    elif cv_gener and not sliding_window and n_splits == None and not incr_train_data and not \
            learn_pr_incr:
        Y_val_pred, Y_test_pred, trained_regrs = _linear_regression_cv_gener(X_tr, X_val, \
            X_test, Y_train, Y_val, Y_test)

    elif incr_train_data and not sliding_window and not cv_regr and not cv_gener and not learn_pr_incr:
        Y_val_pred, Y_test_pred, trained_regrs = _linear_regression_incr_train_data(X_tr, X_val, \
            X_test, Y_train, Y_val, Y_test, regr_type)
    elif learn_pr_incr and not sliding_window and not cv_regr and not cv_gener:
        Y_val_pred, Y_test_pred, trained_regrs = _linear_regression_simple(X_tr, X_val, \
            X_test, Y_train, Y_val, Y_test, regr_type)
    return Y_val_pred, Y_test_pred, trained_regrs


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Train linear regression on train set shared space EEG'
    'and predict shared space of the test set individually for every subject.')
    parser.add_argument('-eeg_dir', type=str, help='Directory with eeg in shared space',\
        default="/scratch/akitaitsev/intersubject_generalizeation/linear/multiviewica/")
    parser.add_argument('-dnn_dir', type=str, help="Directory with DNN activations compliant with Ale's"
    " folder structure", default= '/scratch/akitaitsev/encoding_Ale/')
    parser.add_argument('-out_dir', type=str, help='Output directotry to save predicted shared space EEG')
    parser.add_argument('-is_raw', action='store_true', default=False, help='Bool, raw data flag.')
    parser.add_argument('-regr_type', type=str, default=None, help='average/subjectwise/cv. Whether to perform regression on '
    'data averaged across all subjects, subjectwiese or in cross-validation framework.')
    parser.add_argument('-sliding_window',action='store_true', default=False, help='Bool flag to perform sliding '
    'window regression. If set, fits a separate regression for each sliding window.')
    parser.add_argument('-n_splits', type=int, default=None, help='Number of splits in cross validation.'
    'Note, that cross validation is performed on subject dimension, not on feature dimension!. '
    'Default=None, no cross validation')
    parser.add_argument('-cv_gener',action='store_true', default=False, help='Bool flag whether to perform regression '
    'on the data obtained with cross-validation intersubject generalization. Default=false.')
    parser.add_argument('-cv_regr', action='store_true', default=False, help='Bool flag whether to perform regression '
    'on the data obtained with cross-validation. Default=False.')
    parser.add_argument('-incr_train_data', action='store_true', default=False, help='Flag, whether to use model '
    'paradigm with increment of train data. Default=False')
    parser.add_argument('-learn_pr_incr', action='store_true', default=False, help='Flag, whether to use model '
    'paradigm with learning projections on incremental amout of train data. Default=False')
    args = parser.parse_args()
    ### Might be good idea to get rid of flags and make somethimg like args.paradigm, type=str

    
    # Load shared space eeg data of shape (subj, features, images)
    if bool(args.is_raw):
        filenames=['featurematrix_train.pkl', 'featurematrix_val.pkl','featurematrix_test.pkl'] 
    elif not bool(args.is_raw):
        filenames = ['shared_train.pkl', 'shared_val.pkl', 'shared_test.pkl']
    try:
        Y_train, Y_val, Y_test = load_intersubject_eeg(args.eeg_dir, filenames)
    except FileNotFoundError:
        print('No such method: '+ str(args.eeg_dir))
        sys.exit(0)    
    
    # Argument compatibilibty check
    if args.n_splits != None and args.n_splits > Y_train.shape[0]:
        raise ValueError("Number of cross validation spkits is larger than the number of subjects!")
    elif args.n_splits != None and Y_train.shape[0]%args.n_splits!=0:
        warnings.warn("Number of subject is not integer dividible into number of splits!")

    # Load dnn activations
    X_tr, X_val, X_test = load_dnn_data('CORnet-S', 1000, args.dnn_dir)

    # linear regression    
    Y_val_pred,Y_test_pred, trained_regrs = linear_regression(X_tr, X_val, \
        X_test, Y_train, Y_val, Y_test, args.regr_type, args.sliding_window, args.n_splits, \
        args.cv_regr, args.cv_gener, args.incr_train_data, args.learn_pr_incr)

    # save predicted eeg
    none2str = lambda x: str('') if x == None else x
    path=Path(args.out_dir)
    if not path.is_dir():
        path.mkdir(parents=True)
    joblib.dump(Y_test_pred, (path/('Y_test_predicted_'+none2str(args.regr_type)+'.pkl')))    
    joblib.dump(Y_val_pred, (path/('Y_val_predicted_'+none2str(args.regr_type)+'.pkl')))
    joblib.dump(trained_regrs, (path/('trained_regression_'+none2str(args.regr_type)+'.pkl')))
