'''Create plots to represent sliding window generic
import joblib
decoding accuracy on multiviewica with PCA '''
#! /bin/bash
from pathlib import Path
from analysis_utils import cget_data_sliding_window, reate_time_axis, topplots_sliding_window
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-save_fig', action='store_true', default=False, help='Flag, whether to save '
    'figure. Default=False.')
parser.add_argument('-save_data', action='store_true', default=False, help='Flag, whether to save '
    'top N accuracies data. Default=False.')
parser.add_argument('-out_dir', type=str, 
    default='/scratch/akitaitsev/intersubject_generalization/results/linear/gridsearches/',
    help='Directory to save the data and figure. Default= '
    '/scratch/akitaitsev/intersubject_generalization/results/linear/gridsearches/')
parser.add_argument('-fig_name', type=str, 
    default='time_resolution-top1-pca10_pca50.png', help='Figure name. Default=time_resolution-top1-pca10_pca50.png')
parser.add_argument('-data_name', type=str, 
    default='time_resolution-top1-pca10_pca50.csv', help='Data name. Default=time_resolution-top1-pca10_pca50.csv')

args = parser.parse_args()

# Topplots PCA 100hz (time-window 16-80)
sw_fname='generic_decoding_results_subjectwise.pkl'
av_fname='generic_decoding_results_average.pkl'
filepaths=['/scratch/akitaitsev/intersubject_generalization/linear/dataset1/generic_decoding/sliding_window/multiviewica/main/pca/10/100hz/time_window16-80/',\
    '/scratch/akitaitsev/intersubject_generalization/linear/dataset1/generic_decoding/sliding_window/multiviewica/main/pca/50/100hz/time_window16-80/']
labels1 = ['PCA10', 'PCA50']


# get timecourses data 
av_time, sw_time, sw_time_sd =  get_data_sliding_window(av_fname, sw_fname, filepaths)

# plot figures
timepoints = create_time_axis(5, 16, 80, 100) # win_len=5 samples, start=16 samples, end=80 samples, sr=100Hz)
fig1, ax1 = topplots_sliding_window(av_time, sw_time, sw_time_sd, labels1, top=1, timepoints=timepoints, \
    title='Top 1 generic decoding accuracy for sliding time window, multiviewica')

# save
if rags.save_fig or args.save_data:
    out_path1 = Path(args.save_fig_path)
    if not out_path1.is_dir():
        out_path1.mkdir(parents=True)

if args.save_fig:
    fig1.savefig(out_path1.joinpath('top1-pca10_pca50.png'), dpi=300)
if args.save_data:    
    data = pd.DataFrame(np.array((av_time, sw_time, sw_time_sd)),\
        columns = labels1, index=['av_timecourse','sw_timecourse_mean','sw_timecourse_sd'])
    data.to_csv(out_dir.joinpath(args.data_name))

# Topplots srm 16-80
#    sw_fname='generic_decoding_results_subjectwise.pkl'
#    av_fname='generic_decoding_results_average.pkl'
#    filepaths=['/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/sliding_window/multiviewica/main/srm/10/100hz/time_window16-80/',\
#        '/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/sliding_window/multiviewica/main/srm/50/100hz/time_window16-80/']
#    labels2 = ['srm10', 'srm50']
#    timepoints = create_time_axis(5, 16, 80, 100) # wind len, wind begin, wind end, sr
#    fig2, ax2 = topplots_sliding_window(av_fname, sw_fname, filepaths, labels2, top=1,timepoints=timepoints, \
#        title='Generic decoding top 1 results per time window for multiviewica')
#    out_path2=Path('/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/plots/sliding_window/multiviewica/100hz/time_window16-80/')
#    if not out_path2.is_dir():
#        out_path2.mkdir(parents=True)
#    fig2.savefig(out_path2.joinpath('top1-srm10_srm50.png'), dpi=300)
    plt.show()
