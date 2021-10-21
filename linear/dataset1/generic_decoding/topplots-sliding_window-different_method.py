#! /bin/bash
'''Create plots to represent sliding window generic decoding accuracy for different methods'''
from pathlib import Path
from analysis_utils_linear import get_data_sliding_window, create_time_axis, topplots_sliding_window
import pandas as pd
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('-save_fig', action='store_true', default=False, help='Flag, whether to save '
    'figure. Default=False.')
parser.add_argument('-save_data', action='store_true', default=False, help='Flag, whether to save '
    'top N accuracies data. Default=False.')
parser.add_argument('-out_dir', type=str,default='/scratch/akitaitsev/intersubject_generalization/results/linear/',
    help='Directory to save the data and figure. Default=/scratch/akitaitsev/intersubject_generalization/results/linear/')
parser.add_argument('-fig_name', type=str, 
    default='sliding_window-different_methods.png', help='Figure name. Default=sliding_window-different_methods.png')
parser.add_argument('-data_name', type=str, default='sliding_window-different_methods.csv', help='Data name.\
    Default=sliding_window-different_methods.csv')

args = parser.parse_args()

sw_fname='generic_decoding_results_subjectwise.pkl'
av_fname='generic_decoding_results_average.pkl'
fpath_base='/scratch/akitaitsev/intersubject_generalization/linear/dataset1/generic_decoding/\
sliding_window-different_methods/'

fpaths=[]
methods=['multiviewica', 'permica', 'groupica', 'control']
linestyles=['solid', 'solid', 'solid', 'dashed']
for method in methods:
    fpaths.append((fpath_base+method+'/100hz/'))

# get timecourses data 
av_time, sw_time, sw_time_sd =  get_data_sliding_window(av_fname, sw_fname, fpaths)

# plot figures
timepoints = create_time_axis(5, 26, 80, 100) # win_len=5 samples, start=26 samples, end=80 samples, sr=100Hz)
fig1, ax1 = topplots_sliding_window(av_time, sw_time, sw_time_sd, methods, top=1, timepoints=timepoints, \
    title='Top 1 generic decoding accuracy for sliding time window', linestyles=linestyles)

# save
if args.save_fig or args.save_data:
    out_path1 = Path(args.save_fig_path)
    if not out_path1.is_dir():
        out_path1.mkdir(parents=True)

if args.save_fig:
    fig1.savefig(out_path1.joinpath('sliding_window-different_methods.png'), dpi=300)
if args.save_data:    
    data = pd.DataFrame(np.array((av_time, sw_time, sw_time_sd)),\
        columns = labels1, index=['av_timecourse','sw_timecourse_mean','sw_timecourse_sd'])
    data.to_csv(out_dir.joinpath(args.data_name))
plt.show()

