import numpy as np
import joblib
from pathlib import Path


class Preprocessor:
    def __init__(self,
        data_dir,
        time_window = None
        ):
        '''
        Inputs:
            time_window - list, time interval after the stimulus presentation to select.
                Default None - select all timepoints.
            data_dir - str, root directory with eeg data
        '''
        self.data_dir = Path(data_dir)
        self.time_window = time_window
    
    def create_dataset_matrix(self,
        dataset,
        subjects = None,
        ):
        '''
        Function packs datasets from single subject and session
        into a single matrix.
        Note, that train and test data have different formats:
            train (im, rep, ch, time)
            test (im, rep, ch, time)

        Inputs:
            dataset - str, type of dataset (train, test)
            data_dir - str, root directory with eeg data
            subjects - list/tuple of str of subject numbers in form (01,02,etc.).
                        Default = None (then uses all subjs
                        [01, 02, 03, 04, 05, 06, 07]) 
        Outputs:
            data_cum - numpy array of shape  (n_subjs, n_images, n_channels, n_times) 
        '''
        if subjects == None:
            subjects = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')
        subjects = ['sub-'+el for el in subjects]

        data_cum = []
        
        # Loop through subject folders
        for subj in subjects:
            
            data_subj = np.load(self.data_dir.joinpath('{}/preprocessed_eeg_{}.npy'.
                format(subj, dataset)), allow_pickle=True).item()
            data_subj_eeg = data_subj["preprocessed_eeg_data"]

            # average EEG across stimulus repetitions
            data_subj_eeg = np.mean(data_subj_eeg, axis=1)

            # select time window
            if not self.time_window is None:
                time_start = np.where(data_subj["times"] == self.time_window[0])[0]
                time_end = np.where(data_subj["times"] == self.time_window[1])[0]
                data_subj_eeg = data_subj_eeg["preprocessed_eeg_data"][:, :, time_start:time_end]

            # dat is now of shape (images, channels, times)
            data_cum.append(data_subj_eeg)

        data_cum = np.stack(data_cum, axis=0) # shape (subjects,images, channels, times)
        
        return data_cum
    
    def dataset_to_feature_matrix(self, dataset):
        '''
        Reshapes dataset of shape (subj, im, ch, times) 
        into the format suitable for multiviewica of shape
        (subjs, chans*times, images)
        Inputs:
            dataset - nd numpy array of shape (subj, im, chans, time)
        Output:
            feature_matrix - np.array of shape 
                (subjs, chans*times=features, images)
        '''
        subj, im, ch, time= dataset.shape
        dataset = np.transpose(dataset, (0, 2, 3, 1)) # (subjects,images, channels, times)
        dataset = np.reshape(dataset, (subj, -1, im)) # (subjs, chans*times, images)
        return dataset

    def preprocess(self):
        '''
        Read reprocessed EEG data for different subjects and convert it
        into numpy array of feature matrices for trainand test datasets.
        Returns: 
            (featrematrix_train, featrematrix_test) tupe of np arays of shape
                (subjs, chans*times, images)
        '''
        featurematrices = []
        for dataset in ("training", "test"):
            data_mat = self.create_dataset_matrix(dataset)
            featurematrices.append(self.dataset_to_feature_matrix(data_mat))
        return featurematrices
