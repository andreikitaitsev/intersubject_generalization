#! /bin/bash/python3

import joblib
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def scan_ncomp_from_dir(ddir, regions=None):
    '''Scan the number of components for each region and return sorted list 
    of ncomp for each region.
    Inputs:
        ddir - root directory where teh data is stored
        regions - list of regions to use, default=None, uses all the regions.
    Outputs:
        comps - dict containing the lists of ordered number of components for 
            each region
    '''
    if regions is None:
        regions=('WB', 'EBA', 'FFA', 'LOC', 'PPA', 'STS', 'V1', 'V2', 'V3', 'V4')
    ddir = Path(ddir)
    comps=defaultdict()
    for reg in regions: 
        tmpdir = ddir.joinpath(reg)
        tmp_lst = [el.parts[-1][3:] for el in tmpdir.iterdir() if el.is_dir() and 'PCA' in str(el)]
        comps[reg]=list(map(str, sorted(list(map(int, tmp_lst)))))
    return comps

def load_dnn_activations(base_dir, dataset, layer):
    '''Load extracted DNN features (activations) for
    train or test dataset.
    Inputs:
        base_dir - str or Path-like object.
        dataset - str, train or test
        layer - str, dnn layer to use
    Outputs:
        acts - 2d np array of dnn features.
    '''
    base_dir = Path(base_dir)
    acts = joblib.load(base_dir.joinpath(layer+'_'+(str(dataset)+'_activations.pkl')))
    return acts

def load_ssr_fmri(base_dir, region, dataset):
    '''
    Load shared space fMRI response
    and transpose it from (subj, features, n_videos) to (subj, n_videos, features)
    Inputs:
        base_dir - str or Path-like object.
        region - str, brain region 
        dataset - str, train or test
    Outputs:
        data - 3d numpy array (subj, n_videos, features)
    '''
    data = joblib.load(Path(base_dir).joinpath((str(region)+'_shared_'+dataset+'.pkl')))
    data = np.transpose(data, (0, 2, 1))
    return data

class Linear_Regression(object):
    ''' Object to perform regression.
    Accepts arbitrary regression object and preprocessor.
    Parameters:
        regression - object to perform linear regression. Must implement 
        .fit and .predict methods. Default - linear regression
        preprocessor - object to preprocess video activations before regression.
        Must implement .fit and .transform methods. Default - Standard Scaler.
    Methods:
        .call - performs regression and returns Y_te_predicted.
    '''
    def __init__(self, regression=None, preprocessor=None):
        if regression is None:
            self.regression = LinearRegression()
        else:
            self.regression = regression
        if preprocessor is None:
            self.preprocessor = StandardScaler()
        else:
            self.preprocessor = preprocessor

    def __call__(self, X_tr, X_te, Y_tr, Y_te):
        ''' Fit linear regression object on train data and use it to predict test data.
        Return predicted average and subjectwise data
        '''
        self.preprocessor.fit(X_tr)
        X_tr_trans = self.preprocessor.transform(X_tr) 
        X_te_trans = self.preprocessor.transform(X_te) 
        # average
        mean=lambda x: np.mean(x, axis=0)
        self.regression.fit(X_tr_trans, mean(Y_tr))
        Y_te_pred_av = self.regression.predict(X_te_trans)
        # subject-wise
        Y_te_pred_sw=[]
        for subj in range(Y_tr.shape[0]):
            self.regression.fit(X_tr_trans, Y_tr[subj,:,:])
            Y_te_pred_sw.append(self.regression.predict(X_te_trans))
        Y_te_pred_sw=np.stack(Y_te_pred_sw, axis=0)
        return Y_te_pred_av, Y_te_pred_sw

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Perform linear regression with standard '
    'scaler preprocessing.')
    parser.add_argument('-x_dir', '--dnn_dir', type=str, help='Directory for dnn activations.')
    parser.add_argument('-y_dir', '--fmri_dir', type=str, help='Directory for fMRI responses. If '
    'auto_scan, the region and PCAncomp subdirs will be automatically inferred.')
    parser.add_argument('-region', type=str, help='Brain region to use. If WB - uses whole brain data '
    'full_track in ALgonautus. Other regions available from mini_track: EBA, FFA, LOC, PPA, STS, V1, V2, V3, V4.')
    parser.add_argument('-layers', type=str,nargs='+', help='List of DNN layers to use.')
    parser.add_argument('-out_dir', type=str, help='Output direcotry to save predicted fMRI test data. '
    'If auto_scan, region and PCAncomp subdirs will be automatically inferred and created.')
    parser.add_argument('-auto_scan', action='store_true', default=False, help='Flag, whether to auto detect '
    'number of components from the root dir. Useful when user did not explicitely choose the n of components '
    '(no need to manually look up for them). Changes y_dir and out_dir accordingly. Default=False.')
    args = parser.parse_args()
    regression = Linear_Regression()

    for layer in args.layers:
        X_tr = load_dnn_activations(args.dnn_dir, 'train', layer)
        X_te = load_dnn_activations(args.dnn_dir, 'test', layer)

        if args.auto_scan:
            comps=scan_ncomp_from_dir(args.fmri_dir, [args.region])
            for ncomp in comps[args.region]:
                ydir_it = Path(args.fmri_dir).joinpath(args.region, ('PCA'+str(ncomp)))
                Y_tr = load_ssr_fmri(ydir_it, args.region, 'train')
                Y_te = load_ssr_fmri(ydir_it, args.region, 'test')
                Y_te_pred_av, Y_te_pred_sw = regression(X_tr, X_te, Y_tr, Y_te)
                out_dir_it = Path(args.out_dir).joinpath(args.region, ('PCA'+str(ncomp)))
                if not out_dir_it.is_dir():
                    out_dir_it.mkdir(parents=True)
                joblib.dump(Y_te_pred_av, out_dir_it.joinpath((args.region+'_test_pred_av.pkl')))
                joblib.dump(Y_te_pred_sw, out_dir_it.joinpath((args.region+'_test_pred_sw.pkl')))
                
        elif not args.auto_scan:
            Y_tr = load_ssr_fmri(args.fmri_dir, args.region, 'train')
            Y_te = load_ssr_fmri(args.fmri_dir, args.region, 'test')

            Y_te_pred_av, Y_te_pred_sw = regression(X_tr, X_te, Y_tr, Y_te)

            out_dir = Path(args.out_dir)
            if not out_dir.is_dir():
                out_dir.mkdir(parents=True)
            joblib.dump(Y_te_pred_av, out_dir.joinpath((layer+'_'+args.region+'_test_pred_av.pkl')))
            joblib.dump(Y_te_pred_sw, out_dir.joinpath((layer+'_'+args.region+'_test_pred_sw.pkl')))
