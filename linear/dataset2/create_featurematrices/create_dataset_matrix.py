#! /bin/env/python
'''
Script to convert data from different subjects and sessions into
a single dataset matrix.
'''
import numpy as np
import joblib
from pathlib import Path

def create_dataset_matrix(dataset, data_dir, srate=50, subjects = None, \
    av_reps=True):
    '''
    Function packs datasets from single subject and session
    into a single matrix.
    Note, that in dataset2 train and test data have different formats:
        train (im, rep, ch, time)
        test (sess, im, rep, ch, time)

    Inputs:
        dataset - str, type of dataset (train, val, test)
        data_dir - str, root directory with eeg data
        srate - int, sampling rate of eeg to use.Default=50.
        subjects - list/tuple of str of subject numbers in form (01,02,etc.).
                    Default = None (then uses all subjs
                    [01, 02, 03, 04, 05, 06, 07]) 
        av_reps - average_repetitions - bool, whether to average over repetitions.
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
        subjects = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')
    subjects = ['sub-'+el for el in subjects]

    data_subj = []
    # Loop through subject folders
    path = Path(data_dir)
    for subj in subjects:
        dat = np.load(path.joinpath(subj, 'eeg', ('dtype-'+dataset),\
            ('hz-'+'{0:04d}'.format(srate)),'preprocessed_data.npy'), allow_pickle=True).item()
        dat = dat["prepr_data"]
        if dataset == "test": #shape (sess, im, rep, ch, time)
            dat = np.transpose(dat, (1,0,2,3,4)) # shape (im, sess, rep, ch, time)
            im, sess, rep, ch, time = dat.shape
            dat = np.reshape(dat, (im, sess*rep, ch, time)) 
            # to shape (im, rep, ch, time), where reps are from different sessions
            inds = np.random.permutation(np.linspace(0, sess*rep, sess*rep, endpoint=False, dtype=int))
            # repetitions randomly come from diff sessions 
            dat = dat[:, inds, :, :] # (im, new_rep, ch, time)
        elif dataset == "train":
            pass
        # dat is now of shape (ims, reps, chs, times)
        data_subj.append(dat)
    data = np.stack(data_subj, axis=0) # shape (subj, im, reps, chans, times)
    if av_reps:
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
    parser.add_argument('-omit_val',type=bool, default=True, help='whether to omit validation set. Dataset2 does not '
    'have val dataset.')
    parser.add_argument('-av_reps',type=bool, default=True, help='whether to average across repetitions. If true, '
    'dataset.shape = (subj, im, ch, time), if False dataset.shape = (subj, im, reps, ch, time). Default=True.')
    args = parser.parse_args() 

    # create train, val and test datasets
    dataset_train = create_dataset_matrix('train', args.input_dir, srate=args.srate, av_reps=args.av_reps)
    dataset_test = create_dataset_matrix('test', args.input_dir, srate=args.srate, av_reps=args.av_reps)
    if not args.omit_val: 
        dataset_val = create_dataset_matrix('val', args.input_dir, srate=args.srate)
    # check if specific time windows shall be used
    if args.time_window != None:
        dataset_train = dataset_train[:,:,:,args.time_window[0]:args.time_window[1]+1 ]
        dataset_test = dataset_test[:,:,:,args.time_window[0]:args.time_window[1]+1 ]
        if not args.omit_val:
            dataset_val = dataset_val[:,:,:,args.time_window[0]:args.time_window[1]+1 ]
    
    # save dataset matrices
    out_dir=Path(args.output_dir)
    if not out_dir.is_dir(): 
        out_dir.mkdir(parents=True) 

    joblib.dump(dataset_train, out_dir.joinpath('dataset_train.pkl'))
    joblib.dump(dataset_test, out_dir.joinpath('dataset_test.pkl'))
    if not args.omit_val: 
        joblib.dump(dataset_val, out_dir.joinpath('dataset_val.pkl'))
