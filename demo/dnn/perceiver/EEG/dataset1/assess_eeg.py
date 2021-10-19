#! /bin/env/python3

import numpy as np
import os
from pathlib import Path
import joblib
import copy
import sys
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression   
from sklearn.preprocessing import StandardScaler


# Load data
def load_dnn_data(net, n_pca, project_dir):
    """Loading the DNN activations of training, validation and test data.
    Parameters
    ----------
    net : str
            Used DNN net.
    n_pca : int
            PCA downsampling dimensionality of DNN activations.
    project_dir : str
            Directory of the project folder.

    Returns
    -------
    DNN activations of training, validation and test data.
    """

    # DNN activations directory ###
    data_dir = "pca_activations/" + net + "/layers-combined/" \
            + "pca_" + format(n_pca, "05")
    file_name = "pca_fmaps.npy"
    # Loading the DNN activations ###
    activations = np.load(os.path.join(project_dir, data_dir, file_name), \
        allow_pickle=True).item()
    X_train = activations["fmaps_train"]
    X_val = activations["fmaps_val"]
    X_test = activations["fmaps_test"]
    return X_train, X_val, X_test

def load_intersubject_eeg(inp_dir, filenames):
    '''Loads the output of linear_intersubject_generalization_utils.py
    Input:
        inp_dir - str, directory where the matrix with intersubject data is stored
        filenames - list of strings of filenames to load
    Output:
        files - list of loaded files
    '''
    files = []
    path = Path(inp_dir)
    for fl in filenames:
        files.append(joblib.load(path.joinpath(fl)))
    return files


# regression
def linear_regression(X_tr, X_test, Y_train, Y_test, regr_type):
    '''Perform linear regression on the whole featurespace.
    Inputs:
        X_tr, X_test - dnn features 
        Y_tr, Y_test - eeg projected into shared space of shape (subj, ims, features)
    Outputs:
        Y_test_pred - predicted test set EEG
        trained_regr - trained regression objects
    '''
    scaler = StandardScaler()
    regr = LinearRegression()
    trained_regrs = []
    if regr_type == 'average':
        Y_train = np.mean(Y_train, axis=0) 
        #Y_val = np.mean(Y_val, axis=0) 
        Y_test = np.mean(Y_test, axis=0) 
        # fit regr
        regr.fit(scaler.fit_transform(X_tr), Y_train) 
        #Y_val_pred = regr.predict(scaler.fit_transform(X_val))
        Y_test_pred= regr.predict(scaler.fit_transform(X_test))
        trained_regrs.append(regr)
    elif regr_type == 'subjectwise':
        #Y_val_pred=[]
        Y_test_pred=[]
        for subj in range(Y_train.shape[0]):
            trained_regrs.append(copy.deepcopy(regr))
            trained_regrs[-1].fit(scaler.fit_transform(X_tr), Y_train[subj])
            #Y_val_pred.append(trained_regrs[-1].predict(scaler.fit_transform(X_val)))
            Y_test_pred.append(trained_regrs[-1].predict(scaler.fit_transform(X_test)))
        #Y_val_pred = np.stack(Y_val_pred, axis=0)
        Y_test_pred = np.stack(Y_test_pred, axis=0)
    return Y_test_pred, trained_regrs


# generic decoding
def generic_decoding(real_data, pred_data, regr_type):
    '''Generic decoding with average or subjectwise regression.
    Inputs:
        real_data - 3d numpy array of shape (subj, features, images)
        pred_data - 3d numpy array of shape (subj, images, features)
        regr_type - str, average or subjectwise
    Outputs:
        cor - list of 2d numpy arrays 
        res - list of 1d numpy arrays of generic decoding results
    '''
    if regr_type == 'subjectwise':
        # correlation matrices between real and predicted EEG resposes for different images
        cor_mat = []
        for subj in range(real_data.shape[0]):
            # matrix of shape (images, images)
            cor_mat_iter = np.zeros((real_data.shape[1],real_data.shape[1]))
            for x in range(cor_mat_iter.shape[0]):
                for y in range(cor_mat_iter.shape[1]):
                    cor_mat_iter[x,y] = np.corrcoef(real_data[subj, x, :],\
                        pred_data[subj, y, :])[0,1]
            cor_mat.append(cor_mat_iter)

        cor_mat = np.array(cor_mat)
        # generic decoding
        gen_dec = []
        for subj in range(real_data.shape[0]):
            gen_dec_it = []
            for im in range(cor_mat.shape[1]):
                ids = np.flip(np.argsort(cor_mat[subj, im, :])) # descending ar
                # get the position of the image in descending correlations row
                gen_dec_it.append((np.where(ids==im)[0][0] +1))
            gen_dec.append(gen_dec_it)

    elif regr_type == 'average':
        # average real data across subjects
        real_data = np.mean(real_data, axis=0)

        # correlation matrices between real and predicted EEG resposes for different images
        cor_mat = np.zeros((real_data.shape[0], real_data.shape[0]))
        for x in range(cor_mat.shape[0]):
            for y in range(cor_mat.shape[1]):
                cor_mat[x,y] = np.corrcoef(real_data[x, :],\
                    pred_data[y,:])[0,1]
        
        # generic decoding 
        gen_dec = []
        for im in range(cor_mat.shape[0]):
            ids = np.flip(np.argsort(cor_mat[im, :])) # descending ar
            # get the position of the image in descending correlations row
            gen_dec.append((np.where(ids==im)[0][0] +1))
        cor_mat = np.array(cor_mat)
    gen_dec = np.array(gen_dec)
    return cor_mat, gen_dec 


def res2hist(fl):
    '''Convert generic decoding results into histograms)
    In the current project data structures, the output files are
    named the same and the paths differ.
    Inputs:
        fl - 2d or 1d np array of generic_decoding_result_average/subjectwise
    Outputs:
        hist - list of arrays of histogram or list of lists of
               arrays of histograms for different subjects
    
    '''
    hist=[]
    if np.ndim(fl) == 2:
        for s in range(fl.shape[0]):
            hist_tmp=np.histogram(fl[s], np.linspace(1,\
                max(set(fl[s]))+1, max(set(fl[s])), endpoint=False, dtype=int))
            hist.append( (hist_tmp[0]/sum(hist_tmp[0]))*100 )
    elif np.ndim(fl) ==1:
        hist_tmp = np.histogram(fl, np.linspace(1,max(set(fl))+1,\
            max(set(fl)), endpoint=False, dtype=int))
        hist.append((hist_tmp[0]/sum(hist_tmp[0]))*100)
    return hist

def hist2top(hist, top, return_sd=False):
    '''Returns the percent of images in top N best correlated images.
    Inputs:
        hist - list of arrays (possibly for different subjects),
               output of res2hist function
        top - int, position of the image (starting from 1!)
        return_sd - bool, whether to return sd over subjects. Default=False
    Returns:
        top_hist - np.float64 
        sd - standard deviation over subjects if hist has len >2
    '''
    sd = None
    top = top-1 # python indexing
    if len(hist) >1:
        top_hist = []
        for s in range(len(hist)):
            if len(np.cumsum(hist[s])) >= top+1: 
                top_hist.append(np.cumsum(hist[s])[top])
            elif len(np.cumsum(hist[s])) < top+1:
                top_hist.append(np.cumsum(hist[s])[-1])
        sd = np.std(np.array(top_hist))
        top_hist = np.mean(top_hist) 
    elif len(hist) ==1:
        cumsum = np.cumsum(hist[0], axis=0)
        if len(cumsum) >= top+1:
            top_hist = cumsum[top]
        elif len(cumsum) < top+1:
            top_hist = cumsum[-1]
    if return_sd:
        return top_hist, sd
    else:
        return top_hist

# pipeline assess EEG projection quality

def assess_eeg(X_train, X_test, Y_train, Y_test, top=1):
    '''
    Inputs:
        X_train, X_test - 2d numpy arrays of DNN features for regression
        Y_train, Y_test - 3d numpy array of shape (subj, ims, features) of
                  train and test set EEG data projected into shared space
        top - int, default=1
    Outputs:
        top1_av - float, ratio of top 1 predicted images
        top1_av_sd - 
    '''

    # Regression
    Y_pred_av, tr_regr_av = linear_regression(X_train, X_test, Y_train, Y_test,\
        regr_type='average')
    Y_pred_sw, tr_regr_sw = linear_regression(X_train, X_test, Y_train, Y_test,\
        regr_type='subjectwise')

    # Generic decoding
    cor_mat_av, res_av = generic_decoding(Y_test, Y_pred_av, regr_type='average')
    cor_mat_sw, res_sw = generic_decoding(Y_test, Y_pred_sw, regr_type='subjectwise')

    # histograms
    hist_av = res2hist(res_av)
    hist_sw = res2hist(res_sw)
    
    # Top 1 ratio
    top1_av, top1_av_sd = hist2top(hist_av, top=top, return_sd=True)
    top1_sw, top1_sw_sd = hist2top(hist_sw, top=top, return_sd=True)
    return (top1_av, top1_av_sd), (top1_sw, top1_sw_sd)
