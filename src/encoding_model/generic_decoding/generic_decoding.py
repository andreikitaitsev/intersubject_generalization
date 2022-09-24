#! /bin/env/python
'''Script contains functions to run generic decoding on 
intersubject generalized data.'''

import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import LeaveOneOut


class GenercDecoder:
    def __init__(self):
        pass

    def generic_decoding_simple(self, real_data, pred_data, regr_type):
        '''
        Generic decoding on data obtained with all subejcts intersubject generalization
        with average or subjectwise regression 
        (no sliding widnows, no cross-validarion).'''

        # squeeze potential sigleton dim
        real_data = np.squeeze(real_data)
        pred_data = np.squeeze(pred_data)

        if regr_type == 'subjectwise':
            # transpose real data to the same shape as predicted (images, features)
            real_data = np.transpose(real_data, (0, 2, 1))

            # correlation matrices between real and predicted EEG resposes for different images
            cor_mat = []
            for subj in range(real_data.shape[0]):
                # shape (images, images)
                cor_mat_iter = np.zeros(
                    (real_data.shape[1], real_data.shape[1]))

                for x in range(cor_mat_iter.shape[0]):
                    for y in range(cor_mat_iter.shape[1]):
                        cor_mat_iter[x, y] = np.corrcoef(real_data[subj, x, :],
                                                         pred_data[subj, y, :])[0, 1]

                cor_mat.append(cor_mat_iter)
            cor_mat = np.array(cor_mat)

            # generic decoding
            gen_dec = []
            for subj in range(real_data.shape[0]):
                gen_dec_it = []

                for im in range(cor_mat.shape[1]):
                    # descending ar
                    ids = np.flip(np.argsort(cor_mat[subj, im, :]))
                    # get the position of the image in descending correlations row
                    gen_dec_it.append((np.where(ids == im)[0][0] + 1))

                gen_dec.append(gen_dec_it)

        elif regr_type == 'average':

            # average real data across subjects
            real_data = np.mean(real_data, axis=0)
            # transpose real data to the same shape as predicted (images, features)
            real_data = np.transpose(real_data, (1, 0))

            # correlation matrices between real and predicted EEG resposes for different images
            cor_mat = np.zeros((real_data.shape[0], real_data.shape[0]))
            for x in range(cor_mat.shape[0]):
                for y in range(cor_mat.shape[1]):
                    cor_mat[x, y] = np.corrcoef(real_data[x, :],
                                                pred_data[y, :])[0, 1]
            cor_mat = np.array(cor_mat)

            # generic decoding
            gen_dec = []

            for im in range(cor_mat.shape[0]):
                ids = np.flip(np.argsort(cor_mat[im, :]))  # descending ar
                # get the position of the image in descending correlations row
                gen_dec.append((np.where(ids == im)[0][0] + 1))

        gen_dec_cum = gen_dec
        cor_mat_cum = cor_mat

        return cor_mat_cum, gen_dec_cum

    def generic_decoding_sliding_window(self, real_data, pred_data, regr_type):
        '''
        Run generic decoding on data obtained with sliding window intersubject generalization.
        '''
        cor_mat_cum = []
        gen_dec_cum = []
        for real_it, pred_it in zip(real_data, pred_data):

            if regr_type == 'subjectwise':

                # transpose real data to the same shape as predicted (images, features)
                real_it = np.transpose(real_it, (0, 2, 1))

                # correlation matrices between real and predicted EEG resposes for different images
                cor_mat = []
                for subj in range(real_it.shape[0]):

                    # matrix of shape (images, images)
                    cor_mat_iter = np.zeros(
                        (real_it.shape[1], real_it.shape[1]))
                    for x in range(cor_mat_iter.shape[0]):
                        for y in range(cor_mat_iter.shape[1]):
                            cor_mat_iter[x, y] = np.corrcoef(real_it[subj, x, :],
                                                             pred_it[subj, y, :])[0, 1]
                    cor_mat.append(cor_mat_iter)

                cor_mat = np.array(cor_mat)

                # generic decoding
                gen_dec = []
                for subj in range(real_it.shape[0]):

                    gen_dec_it = []

                    for im in range(cor_mat.shape[1]):
                        # descending ar
                        ids = np.flip(np.argsort(cor_mat[subj, im, :]))
                        # get the position of the image in descending correlations row
                        gen_dec_it.append((np.where(ids == im)[0][0] + 1))
                    gen_dec.append(gen_dec_it)

            elif regr_type == 'average':
                # average real data across subjects
                real_it = np.mean(real_it, axis=0)
                # transpose real data to the same shape as predicted (images, features)
                real_it = np.transpose(real_it, (1, 0))

                # correlation matrices between real and predicted EEG resposes for different images
                cor_mat = np.zeros((real_it.shape[0], real_it.shape[0]))
                for x in range(cor_mat.shape[0]):
                    for y in range(cor_mat.shape[1]):
                        cor_mat[x, y] = np.corrcoef(real_it[x, :],
                                                    pred_it[y, :])[0, 1]
                cor_mat = np.array(cor_mat)

                # generic decoding
                gen_dec = []
                for im in range(cor_mat.shape[0]):
                    ids = np.flip(np.argsort(cor_mat[im, :]))  # descending ar
                    # get the position of the image in descending correlations row
                    gen_dec.append((np.where(ids == im)[0][0] + 1))

            cor_mat_cum.append(cor_mat)
            gen_dec_cum.append(gen_dec)
        return cor_mat_cum, gen_dec_cum

    def generic_decoding_cv(self, real_data, predicted_data):
        '''
        Generic decoding on leave-one-out CV data. 
        The predicted data shall be a list of cross-validation splits
        Inputs:
            real_data - 3d numpy array: featurematrix (subj, feat, ims) 
                or shared space response (subj, ims, feat)
            pred_data - list of 2d numpy arrays of predicted EEG data
        Outputs:
            cor_mat_cum
            gen_dec_cum
        '''
        def _cv_split_real_data(real_data):
            '''
            Split real EEG data into the list of leave-one-out CV folds.
            Inputs:
                real_data - 3d np array - featurematrix or shared space response of
                shape (subj, feat, ims)
            Outputs:
                split_data - list of 2d numpy arrays (ims, feat) for every subject.
            '''
            real_data = np.transpose(
                real_data, (0, 2, 1))  # (subj, feat, ims) -> (subj, ims, feat)
            split_data = []
            n_subj = real_data.shape[0]
            inds = np.linspace(0, n_subj, n_subj, endpoint=False, dtype=int)
            for te_idx in inds:
                tr_idx = np.setxor1d(te_idx, inds)
                # subject, whos data was not used to train the regression
                split_data.append(real_data[te_idx, :, :].squeeze())
            return split_data

        # split real data accoring to CV used in regression to get the predicted data
        real_data = _cv_split_real_data(real_data)

        cor_mat_cum = []
        gen_dec_cum = []
        for real_it, pred_it in zip(real_data, pred_data):
            # correlation matrices between real and predicted EEG resposes for different images
            cor_mat = np.zeros((real_it.shape[0], real_it.shape[0]))
            for x in range(cor_mat.shape[0]):
                for y in range(cor_mat.shape[1]):
                    cor_mat[x, y] = np.corrcoef(real_it[x, :],
                                                pred_it.squeeze()[y, :])[0, 1]

            # generic decoding
            gen_dec = []
            for im in range(cor_mat.shape[0]):
                ids = np.flip(np.argsort(cor_mat[im, :]))  # descending ar
                # get the position of the image in descending row of correlations
                gen_dec.append((np.where(ids == im)[0][0] + 1))
            cor_mat_cum.append(cor_mat)
            gen_dec_cum.append(gen_dec)
        return cor_mat_cum, gen_dec_cum

    def assess(self, real_data, pred_data, regr_type, sliding_window, cv):
        '''
        Perform generic decoding.
        Inputs:
            real_data - np array of real EEG data
            pred_data - np array of predicted EEG data
            regr_type - str, type of the regression (average, subjectwise)
        Flags - for data comming from the different types of analysis:
            sliding_window 
            cv
        Outputs:
            cor - list of 2d numpy arrays 
            res - list of 1d numpy arrays of generic decoding results
            metadata - dictionary with parameters used
        '''
        if sliding_window and not cv:
            cor, res = self.generic_decoding_sliding_window(
                real_data, pred_data, regr_type)
        elif cv and not sliding_window:
            cor, res = self.generic_decoding_cv(real_data, pred_data)
        elif not sliding_window and not cv:
            cor, res = self.generic_decoding_simple(
                real_data, pred_data, regr_type)
        
        metadata = {'regr_type': regr_type, 'sliding_window': sliding_window}
        
        return cor, res, metadata


    def plot_correlaiton(self, cor):
        # average
        vmin = np.amin(cor)
        vmax = np.amax(cor)
        fig_av, ax = plt.subplots(figsize=(16,9)) 
        im = ax.imshow(cor, origin='lower',aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title('Average shared space correlation matrix')
        cbar = fig_av.colorbar(im, ax=ax)
        cbar.set_label('Correlation')

        # subectwise
        vmin = np.amin(cor)
        vmax = np.amax(cor)
        fig_sw, axes = plt.subplots(4,2, sharex=True, sharey=True, figsize=(16,9)) #7 subjs
        for subj, ax in zip(range(7), axes.flat):
            im = ax.imshow(cor[subj], origin='lower',aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_title('Subject '+str(subj))
        cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
        cbar=plt.colorbar(im, cax=cax, **kw)
        cbar.set_label('Correlation')
        
        return fig_av, fig_sw
    
    def plot_decoding(self, gen_dec_res_av, gen_dec_res_sw):
        '''
        Generic decoding histograms for average and
        subject-wise setup.
        '''
        # average
        unique = set(gen_dec_res_av)
        n_el = len(gen_dec_res_av)
        unique_counter = [ np.sum([gen_dec_res_av[num] == un_el for num in 
            range(len(gen_dec_res_av))])/n_el*100 for un_el in unique]
        fig_av, ax = plt.subplots(figsize=(16,9))
        ax.bar(list(unique), unique_counter)
        ax.set_xlabel('Position of true image among n best correlated images.')
        ax.set_ylabel('Percent ratio')
        fig_av.suptitle('Generic decoding results for shared space averaged between subjects')

        # subjecwise
        fig_sw, axes = plt.subplots(4,2, figsize=(16,9))
        for subj, ax in zip(range(7), axes.flat):
            res_it = gen_dec_res_sw[subj,:]
            unique = set(res_it)
            n_el = len(res_it)
            unique_counter = [ np.sum([res_it[num] == un_el for num in range(len(res_it))])
                /n_el*100 for un_el in unique]
            plot = ax.bar(list(unique), unique_counter)
            ax.set_xlabel('Position of true image among n best correlated images.')
            ax.set_ylabel('Percent ratio.')
            ax.set_title('Subject '+str(subj)) 
        fig_sw.suptitle('Generic decoding results for subjectwise shared space.')
        
        return fig_av, fig_sw





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Runs generic decoding and saves '
                                     'the data and correlation matrices.')
    parser.add_argument('-real', '--real_eeg_filepath', type=str, help='Path to '
                        'the real test set EEG')
    parser.add_argument('-pred', '--pred_eeg_filepath', type=str, help='Path to '
                        'the predicted test set EEG')
    parser.add_argument('-regr_type', type=str, default=None, help="Type of shared space input data"
                        " ('subjectwise' or 'average' between subjects). Default=None.")
    parser.add_argument('-out', '--output_dir', type=str, help='Directory to save '
                        'decoding results and correlation matrices.')
    parser.add_argument('-sliding_window', action='store_true', default=False,
                        help='Bool flag, perform generic decoding on sliding window intersubject generalizeation.')
    parser.add_argument('-cv', action='store_true', default=False,
                        help='Bool flag, perform generic decoding on cross-validated intersubject generalizeation.')
    args = parser.parse_args()

    # Load data
    real_data = joblib.load(args.real_eeg_filepath)
    pred_data = joblib.load(args.pred_eeg_filepath)

    cor, res, metadata = generic_decoding(real_data, pred_data, args.regr_type, args.sliding_window,
                                          args.cv)

    # save results
    def none2str(x): return '' if x == None else str(x)
    path = Path(args.output_dir)
    if not path.is_dir():
        path.mkdir(parents=True)
    joblib.dump(cor, (path / ('generic_decoding_correlations_'
                              + none2str(args.regr_type)+'.pkl')))
    joblib.dump(res, (path / ('generic_decoding_results_'
                              + none2str(args.regr_type)+'.pkl')))
    joblib.dump(metadata, path.joinpath(
        ('metadata'+none2str(args.regr_type)+'.pkl')))
