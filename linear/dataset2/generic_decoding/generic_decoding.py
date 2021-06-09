#! /bin/env/python
'''Script contains functions to run generic decoding on 
intersubject generalized data.'''

import numpy as np
import joblib 
from pathlib import Path


def generic_decoding_simple(real_data, pred_data, regr_type):
    '''Generic decoding on data obtained with all subejcts intersubject generalization
    with average or subjectwise regression (no sliding widnows, no cross-validarion).'''
    # squeeze potential sigle dim
    real_data=np.squeeze(real_data)
    pred_data=np.squeeze(pred_data)
    if regr_type == 'subjectwise':
        # transpose real data to the same shape as predicted (images, features)
        real_data = np.transpose(real_data, (0,2,1))
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
    gen_dec_cum=gen_dec
    cor_mat_cum=cor_mat
    return cor_mat_cum, gen_dec_cum 


if __name__ =='__main__':
    import argparse 
    parser = argparse.ArgumentParser(description='Runs generic decoding and saves '
    'the data and correlation matrices.')
    parser.add_argument('-real','--real_eeg_filepath', type=str, help='Path to '
    'the real test set EEG')
    parser.add_argument('-pred','--pred_eeg_filepath', type=str, help='Path to '
    'the predicted test set EEG')
    parser.add_argument('-regr_type', type=str, default=None, help="Type of shared space input data"
    " ('subjectwise' or 'average' between subjects). Default=None.")
    parser.add_argument('-out','--output_dir', type=str, help='Directory to save '
    'decoding results and correlation matrices.')
    args = parser.parse_args()

    # Load data
    real_data = joblib.load(args.real_eeg_filepath)
    pred_data = joblib.load(args.pred_eeg_filepath)
    
    cor, res = generic_decoding_simple(real_data, pred_data, args.regr_type)

    # save results
    none2str = lambda x: '' if x==None else str(x)
    path=Path(args.output_dir)
    if not path.is_dir():
        path.mkdir(parents=True)
    joblib.dump(cor, (path /('generic_decoding_correlations_'\
        +none2str(args.regr_type)+'.pkl'))) 
    joblib.dump(res, (path /('generic_decoding_results_'\
        +none2str(args.regr_type)+'.pkl')))
