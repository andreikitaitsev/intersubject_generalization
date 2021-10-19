#! /bin/env/python
'''
Script to convert data from different subjects and sessions into
a single dataset matrix.
'''
import numpy as np
import joblib
from pathlib import Path

def create_dataset_matrix(dataset, data_dir, srate=50, subjects = None, sessions=None,\
    average_repetitions=True):
    '''
    Function packs datasets from single subject and session
    into a single matrix and (possibly) averages over repetitions.
    Inputs:
        dataset - str, type of dataset (train, val, test)
        data_dir - str, root directory with eeg data
        srate - int, sampling rate of eeg to use.Default=50.
        subjects - list/tuple of str of subject numbers in form (01,02,etc.).
                    Default = None (then uses all subjs
                    [01, 02, 03, 04, 05, 06, 07]) 
        sessions - list/tuple of str of sessions to use.
                    Default = None, then uses following sessions:
                    [01, 02, 03]
        average_repetitions - bool, whether to average over repetitions.
                              Default = True
    Outputs:
        packed_dataset - nd numpy array of shape 
        
        Default(if average_repetitions == True):
            (n_subjs, n_images, n_channels, n_times) 
        if average_repetitions == False
            (n_subjs, n_images, n_repetitions, n_channels, n_times)
        
    Note, that as different subjects have different number of sessions, 
    sessions are averaged
    '''
    if not (dataset=='train' or dataset=='val' or dataset=='test'):
        raise ValueError("Invalid dataset name!")
    if subjects == None:
        subjects = ('01', '02', '03', '04', '05', '06', '07')
    subjects = ['sub-'+el for el in subjects]
    if sessions == None:
        sessions = ('01',  '02', '03')
    sessions = ['ses-'+el for el in sessions]

    data_subj = []
    # Loop through subject folders
    path = Path(data_dir)
    for subj in subjects:
        data_ses = []
        for ses in sessions:
            try:
                dat = np.load(path.joinpath(subj,ses,'eeg', ('dtype-'+dataset),\
                    ('hz-'+'{0:04d}'.format(srate)),'preprocessed_data.npy'), allow_pickle=True).item()
                dat = dat["prepr_data"]
                data_ses.append(dat)
            except FileNotFoundError:
                print('Subject '+subj+' missing session '+ses)        
        av = lambda x: np.mean(x, axis = 0)
        # shape (ims, reps, chs, times)
        data_subj.append( av(np.stack(data_ses, axis=0)))
    data = np.stack(data_subj, axis=0) # shape (subj, im, reps, chans, times)
    if average_repetitions:
        data = np.mean(data, axis = 2) # shape (subj, im, chans, times)
    return data

if __name__=='__main__': 
    import argparse 
    parser = argparse.ArgumentParser(description='Create dataset matrix from eeg '
    'features for train, test and validation sets.')
    parser.add_argument('-inp', '--input_dir', type=str, help='Root directory of eeg data') 
    parser.add_argument('-out','--output_dir', type=str, help='Directory to save created '
    'dataset matrices.')
    parser.add_argument('-time','--time_window', type=int, nargs=2, help = 'Specific time window to use in the analysis.'
        '2 integers - first and last SAMPLES of the window INCLUSIVELY.', default = None)
    parser.add_argument('-srate', type=int, default=50, help='sampling rate of EEG to load. Default=50.')
    args = parser.parse_args() 

    # create train, val and test datasets
    dataset_train = create_dataset_matrix('train', args.input_dir, srate=args.srate)
    dataset_val = create_dataset_matrix('val', args.input_dir, srate=args.srate)
    dataset_test = create_dataset_matrix('test', args.input_dir, srate=args.srate)
    
    # check if specific time windows shall be used
    if args.time_window != None:
        if not len(dataset_train.shape) == 4:
            raise ValueError("Dataset has 5 dimensions. Presumably, you did not average "
                "over image repetitions.")
        dataset_train = dataset_train[:,:,:,args.time_window[0]:args.time_window[1]+1 ]
        dataset_val = dataset_val[:,:,:,args.time_window[0]:args.time_window[1]+1 ]
        dataset_test = dataset_test[:,:,:,args.time_window[0]:args.time_window[1]+1 ]
    
    # save dataset matrices
    out_dir=Path(args.output_dir)
    if not out_dir.is_dir(): 
        out_dir.mkdir(parents=True) 

    joblib.dump(dataset_train, out_dir.joinpath('dataset_train.pkl'))
    joblib.dump(dataset_val, out_dir.joinpath('dataset_val.pkl'))
    joblib.dump(dataset_test, out_dir.joinpath('dataset_test.pkl'))
