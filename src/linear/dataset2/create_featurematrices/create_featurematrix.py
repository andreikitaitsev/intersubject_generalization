#! /bin/env/python

'''Script to convert dataset matrices into featurematrix 
acceptible for multiviewica (matrix of shape (subjs, features, samples))
'''

import joblib
from pathlib import Path
import numpy as np

__all__ = ['dataset2featurematrix']

def dataset2feature_matrix(dataset):
    '''
    Reshapes dataset of shape (subj, im, rep, ch, times) 
    into the format suitable for multiviewica of shape
    (subjs, chans*times*, images)
    Inputs:
        dataset - nd numpy array of shape (subj, im, chans, time)
                  or (subj, im, reps, chans, time)
    Output:
        feature_matrix - np.array of shape 
            (subjs, features, images) - if dataset has no reps (ndim=4)
            (subj, im, rep, ch, time) - if dataset has reps, pass 
    '''
    # if has reps
    if np.ndim(dataset) == 5: # (subj, im, rep, ch, time) 
        pass
    # if no reps
    elif np.ndim(dataset)==4: # (subj, im, ch, time)
        subj, im, ch, time= dataset.shape
        dataset = np.transpose(dataset, (0, 2,3, 1))
        dataset = np.reshape(dataset, (subj, -1, im))
    return dataset

if __name__=='__main__': 
    import argparse 
    parser = argparse.ArgumentParser(description='Create feature matrix from eeg datasets '
    'for train, test and validation sets (outputs of create_dataset_matrix.py).')
    parser.add_argument('-inp', '--input_dir', type=str, help='EEG datasets directory.') 
    parser.add_argument('-out','--output_dir', type=str, help='Directory to save created featurematrix. '
    'Note, that "/time_window../" directory will be automatically created in the output dir.')
    parser.add_argument('-omit_val', type=bool, default=True, help='Whether to omit val dataset. Dataset2 '
    'does not have val set.')
    args = parser.parse_args() 
    
    # load eeg datasets
    dataset_train = joblib.load(Path(args.input_dir).joinpath('dataset_train.pkl'))
    dataset_test = joblib.load(Path(args.input_dir).joinpath('dataset_test.pkl'))
    if not args.omit_val:
        dataset_val = joblib.load(Path(args.input_dir).joinpath('dataset_val.pkl'))
    
    # transform tarin, val and test datasets into feature matrices
    featuremat_train = dataset2feature_matrix(dataset_train)
    featuremat_test =  dataset2feature_matrix(dataset_test)
    if not args.omit_val:
        featuremat_val =  dataset2feature_matrix(dataset_val,args.av_reps)

    # save feature matrices
    out_dir=Path(args.output_dir)
    if not out_dir.is_dir(): 
        out_dir.mkdir(parents=True) 
    joblib.dump(featuremat_train, out_dir.joinpath('featurematrix_train.pkl'))
    joblib.dump(featuremat_test, out_dir.joinpath('featurematrix_test.pkl'))
    if not args.omit_val:
        joblib.dump(featuremat_val, out_dir.joinpath('featurematrix_val.pkl'))
