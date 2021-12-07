#! /bin/env/python
'''Script contains functions to run generic decoding on 
intersubject generalized data.'''

import numpy as np
import joblib 
from pathlib import Path
from sklearn.model_selection import LeaveOneOut
from collections import defaultdict


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

def generic_decoding_fmri(real_data, pred_data, regr_type):
    '''Generic decoding on data obtained with all subejcts intersubject generalization
    with average or subjectwise regression.'''
    def _gen_dec_sw(real_data, pred_data):
        # transpose real data to the same shape as predicted (subj, ims, feat)
        real_data = np.transpose(real_data, (0, 2, 1))
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
        return cor_mat, gen_dec

    def _gen_dec_av(real_data, pred_data):
        # average real data across subjects
        real_data = np.mean(real_data, axis=0)
        # transpose real data to the same shape as predicted (images, features)
        real_data = np.transpose(real_data, (1,0))
        # correlation matrices between real and predicted EEG resposes for different images
        cor_mat = np.zeros((real_data.shape[0], real_data.shape[0]))
        for x in range(cor_mat.shape[0]):
            for y in range(cor_mat.shape[1]):
                cor_mat[x,y] = np.corrcoef(real_data[x, :],\
                    pred_data[y,:])[0,1]
        cor_mat=np.array(cor_mat)
        
        # generic decoding 
        gen_dec = []
        for im in range(cor_mat.shape[0]):
            ids = np.flip(np.argsort(cor_mat[im, :])) # descending ar
            # get the position of the image in descending correlations row
            gen_dec.append((np.where(ids==im)[0][0] +1))
        return cor_mat, gen_dec
    if regr_type == 'average' or regr_type == 'av':
        cor_mat, gen_dec = _gen_dec_av(real_data, pred_data)
    elif regr_type == 'subjectwise' or regr_type=='sw':
        cor_mat, gen_dec = _gen_dec_sw(real_data, pred_data)
    return cor_mat, gen_dec


if __name__ =='__main__':
    import argparse 
    parser = argparse.ArgumentParser(description='Runs generic decoding and saves '
    'the data and correlation matrices.')
    parser.add_argument('-real', type=str, help='Path to '
    'the real test set file. If auto_scan-path to the directory containing data for different regions.')
    parser.add_argument('-pred', type=str, help='Path to '
    'the predicted test set file. If auto_scan-path to the directory containing the data for the different regions.')
    parser.add_argument('-regr_type', type=str, default=None, help="Type of shared space input data"
    " ('subjectwise/sw' or 'average/av' between subjects).")
    parser.add_argument('-out','--output_dir', type=str, help='Directory to save '
    'decoding results and correlation matrices. If auto_scan, will automatically put files in subfolder '
    'PCA ncomp.')
    parser.add_argument('-region', type=str, default=None, help='Region to use. Only use with auto_scan.')
    parser.add_argument('-auto_scan', action='store_true', default=False, help='Flag, whether to auto detect '
    'n components from the root dir. Useful when user did not explicitely choose the n of components '
    '(no need to manually look up for them. Changes y_dir and out_dir accordingly. Default=False.')
    args = parser.parse_args()
    
    if args.auto_scan:
        regr_type_complier=lambda name: 'av' if name == 'av' or name=='average' else 'sw' 
        av_sw_pred_fnames = lambda region, dtype: region+'_test_pred_'+dtype+'.pkl'
        comps=scan_ncomp_from_dir(args.real, [args.region])
        for ncomp in comps[args.region]:
            basedir_real_it = Path(args.real).joinpath(args.region, ('PCA'+str(ncomp)))
            basedir_pred_it = Path(args.pred).joinpath(args.region, ('PCA'+str(ncomp)))
            real_data = joblib.load(basedir_real_it.joinpath((args.region+'_shared_test.pkl')))
            pred_data = joblib.load(basedir_pred_it.joinpath(av_sw_pred_fnames(args.region, \
                regr_type_complier(args.regr_type) ) ) )
            cor, res = generic_decoding_fmri(real_data, pred_data, args.regr_type) 
            # save results
            outdir_it = Path(args.output_dir).joinpath(args.region, ('PCA'+ncomp))
            if not outdir_it.is_dir():
                outdir_it.mkdir(parents=True)
            joblib.dump(cor, outdir_it.joinpath((args.region+'generic_decoding_correlations_'\
                +args.regr_type+'pkl'))) 
            joblib.dump(res, outdir_it.joinpath((args.region+'generic_decoding_results_'\
                +args.regr_type+'.pkl')))

    elif not args.auto_scan:
        # Load data
        real_data = joblib.load(args.real)
        pred_data = joblib.load(args.pred)
        
        cor, res = generic_decoding_fmri(real_data, pred_data, args.regr_type) 

        # save results
        none2str = lambda x: '' if x==None else str(x)
        outdir_it=Path(args.output_dir)
        if not outdir_it.is_dir():
            outdir_it.mkdir(parents=True)
        joblib.dump(cor, outdir_it.joinpath((args.region+'generic_decoding_correlations_'\
            +none2str(args.regr_type)+'.pkl'))) 
        joblib.dump(res, outdir_it.joinpath((args.region+'generic_decoding_results_'\
            +none2str(args.regr_type)+'.pkl')))
