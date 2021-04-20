#! /bin/env/python

'''Script to convert dataset matrices into featurematrix 
acceptible for multiviewica (matrix of shape (subjs, features, samples))
'''

import joblib
from pathlib import Path
import numpy as np

__all__ = ['dataset2featurematrix']

def dataset2feature_matrix(dataset, has_repetitions=False):
    '''
    Reshapes dataset of shape (subj, im, chans,times) is has_repetitions
    ==False or (subj, im, reps, chans, times) if has_repetitions ==True
    into the format suitable for multiviewica of shape
    (subjs, chans*times*, images)
    Inputs:
        dataset - nd numpy array of shape (subj, im, chans, time)
                  or (subj, im, reps, chans, time)
        has_repetitions - bool, whether dataset has repetitions as
                          dimension, i.e. whether average_repetitions=True
                          in create_dataset_matrix func. Default=False
    Output:
        feature_matrix - 3d np.array of shape (subjs, features, images),
                         where features = chans*time if has_repetitions=False
                         or chans*time*reps if has repetitions=True
    '''
    if not has_repetitions:
        dataset = np.transpose(dataset, (0,2,3,1))
    elif has_repetitions:
        dataset = np.transpose(dataset, (0,2,3,4,1))
    dataset = np.reshape(dataset, (dataset.shape[0], -1, dataset.shape[-1]))
    return dataset

if __name__=='__main__': 
    import argparse 
    parser = argparse.ArgumentParser(description='Create feature matrix from eeg datasets '
    'for train, test and validation sets (outputs of create_dataset_matrix.py).')
    parser.add_argument('-inp', '--input_dir', type=str, help='EEG datasets directory.') 
    parser.add_argument('-out','--output_dir', type=str, help='Directory to save created featurematrix. '
    'Note, that "/time_window../" directory will be automatically created in the output dir.')
    args = parser.parse_args() 
    
    # load eeg datasets
    dataset_train = joblib.load(Path(args.input_dir).joinpath('dataset_train.pkl'))
    dataset_val = joblib.load(Path(args.input_dir).joinpath('dataset_val.pkl'))
    dataset_test = joblib.load(Path(args.input_dir).joinpath('dataset_test.pkl'))
    
    # transform tarin, val and test datasets into feature matrices
    featuremat_train = dataset2feature_matrix(dataset_train)
    featuremat_val =  dataset2feature_matrix(dataset_val)
    featuremat_test =  dataset2feature_matrix(dataset_test)

    # save feature matrices
    out_dir=Path(args.output_dir)
    if not out_dir.is_dir(): 
        out_dir.mkdir(parents=True) 
    joblib.dump(featuremat_train, out_dir.joinpath('featurematrix_train.pkl'))
    joblib.dump(featuremat_val, out_dir.joinpath('featurematrix_val.pkl'))
    joblib.dump(featuremat_test, out_dir.joinpath('featurematrix_test.pkl'))
