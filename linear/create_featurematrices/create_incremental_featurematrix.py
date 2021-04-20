#! /bin/env/python3
'''
Create training featurematrices using incremental number of images.
'''
import numpy as np
import joblib
from pathlib import Path
from create_featurematrix import dataset2feature_matrix

def create_incremental_featurematrices(dataset, nsteps):
    '''Creates featurematrices incrementing number of images
    used in each member of sequence. The number of images in divided 
    into nsteps and list of cumsums of data in each interval is returned.
    
    Inputs:
        dataset - 4d numpy array (subj, im, ch, times), output of 
                  create_dataset_matrix function
        nsteps - int, number of steps to divide number of images into. 
                 Corresponds to number of evenly spaced interval to 
                 divide data into.
    Outputs:
        incr_feat - list of 3d numpy array of featurematrices with number
                    of images incremented in every step
    '''
    incr_feat = []
    bins = np.array_split(dataset, nsteps, axis = 1)
    concat = lambda x: np.concatenate(x, axis = 1) if len(x) >1 else x[0] 
    for i in range(1, len(bins)+1):
        incr_feat.append(dataset2feature_matrix(concat(bins[0:i])))
    return incr_feat

if __name__ =='__main__':
    import argparse 
    parser = argparse.ArgumentParser(description='Create feature matrices with incremental ratio of '
    'training images used.')
    parser.add_argument('-nsteps', type=int, help='Number of evenly spaced intervals across image '
    'dimension of the data. Each member of output featurematrices list will be incremented into 1 '
    'such interval.')
    parser.add_argument('-inp', '--input_dir', type=str, help='EEG datasets directory.') 
    parser.add_argument('-out','--output_dir', type=str, help='Directory to save created incremental '
    'train featurematrix. ')
    args = parser.parse_args() 
    
    # load eeg datasets
    dataset_train = joblib.load(Path(args.input_dir).joinpath('dataset_train.pkl'))
    dataset_val = joblib.load(Path(args.input_dir).joinpath('dataset_val.pkl'))
    dataset_test = joblib.load(Path(args.input_dir).joinpath('dataset_test.pkl'))
    
    # create incremental feature matrices for train dataset
    incr_featuremat_train = create_incremental_featurematrices(dataset_train, args.nsteps)
    
    # transform val and test datasets into feature matrices
    featuremat_val =  dataset2feature_matrix(dataset_val)
    featuremat_test =  dataset2feature_matrix(dataset_test)

    # save incremental feature matrices into corrsepodning folder and copy 
    # cal and test featurematrices into each incremental train featuremat folder 
    out_dir=Path(args.output_dir)
    for i in range(len(incr_featuremat_train)):
        outpath=out_dir.joinpath('step_'+str(i))
        if not outpath.is_dir(): 
            outpath.mkdir(parents=True) 
        joblib.dump(incr_featuremat_train[i], outpath.joinpath('featurematrix_train.pkl'))
        joblib.dump(featuremat_val, outpath.joinpath('featurematrix_val.pkl'))
        joblib.dump(featuremat_test, outpath.joinpath('featurematrix_test.pkl'))
