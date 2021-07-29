#! /bin/env/python
from analysis_utils import get_av_sw_data, topplot_av_sw
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-save_fig', action='store_true', default=False, help='Flag, whether to save '
'figure. Default=False.')
parser.add_argument('-save_data', action='store_true', default=False, help='Flag, whether to save '
'top N accuracies data. Default=False.')
parser.add_argument('-out_dir', type=str, 
default='/scratch/akitaitsev/intersubject_generalization/results/linear/control/',
help='Directory to save the data and figure. Default= '
'/scratch/akitaitsev/intersubject_generalization/results/linear/control/')
parser.add_argument('-fig_name', type=str, 
    default='control.png', help='Figure name. Default=control.png')
parser.add_argument('-data_name', type=str, 
    default='control.csv', help='Data name. Default=control.csv')
args = parser.parse_args()

top=1

# filepaths
project_dir = Path('/scratch/akitaitsev/intersubject_generalization/linear/dataset1/',\
    'generic_decoding','control','main')
srate='50hz'
time_window = 'time_window13-40'
av_filename = 'generic_decoding_results_average.pkl'
sw_filename = 'generic_decoding_results_subjectwise.pkl'
methods=['raw','pca/200/']

av_fpaths=[]
sw_fpaths=[]
labels=['raw', 'PCA']
for method in methods:
    av_fpaths.append(project_dir.joinpath(method, srate, time_window, av_filename))
    sw_fpaths.append(project_dir.joinpath(method, srate, time_window, sw_filename))

# get top N data
av_tops, sw_tops, sw_sds = get_av_sw_data(av_fpaths, sw_fpaths, top)

# get top N barplots
fig, axes = topplot_av_sw(av_tops, sw_tops, sw_sds, labels)

# save figure
if args.save_fig or args.save_data: 
    out_dir=Path(args.out_dir)
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True)
if args.save_fig:
    fig.savefig(out_dir.joinpath(args.fig_name), dpi=300)

# save data
if args.save_data:
    import pandas as pd
    data = pd.DataFrame(np.array((av_tops, sw_tops, sw_sds)),\
        columns = labels, index=['av_tops','sw_tops','sw_sds'])
    data.to_csv(out_dir.joinpath(args.data_name))
plt.show()
