#! /bin/env/python
from analysis_utils import get_av_sw_data, topplot_av_sw
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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
    default='gridsearch_over_methods.png', help='Figure name. Default=gridsearch_over_methods.png')
parser.add_argument('-data_name', type=str, 
    default='gridsearch_over_methods.csv', help='Data name. Default=gridsearch_over_methods.csv')
args = parser.parse_args()

top=1

# filepaths
project_dir = Path('/scratch/akitaitsev/intersubject_generalization/linear/dataset1/',\
    'generic_decoding')
srate='50hz'
time_window = 'time_window13-40'
av_filename = 'generic_decoding_results_average.pkl'
sw_filename = 'generic_decoding_results_subjectwise.pkl'
methods=['control/main/raw/','control/main/pca/200/']
av_fpaths=[]
sw_fpaths=[]
labels=['raw', 'PCA']
for method in methods:
    av_fpaths.append(project_dir.joinpath(method, srate, time_window, av_filename))
    sw_fpaths.append(project_dir.joinpath(method, srate, time_window, sw_filename))

# define methods filepaths
add_methods = ['multiviewica', 'groupica', 'permica']
nicks = ['mvica','gica','pica']
preprs = ['pca', 'srm']
for meth, nick in zip(add_methods, nicks):
    for prepr in preprs:
        av_fpaths.append(project_dir.joinpath(meth,'main', prepr, '200', \
            srate, time_window, av_filename))
        sw_fpaths.append(project_dir.joinpath(meth,'main', prepr, '200', \
            srate, time_window, sw_filename))
        labels.append( (prepr+'_'+nick))

# get top N data
av_tops, sw_tops, sw_sds = get_av_sw_data(av_fpaths, sw_fpaths, top)

# get top N barplots
fig, axes = topplot_av_sw(av_tops, sw_tops, sw_sds, labels, hatches=['///', '\\'],\
    xtick_rotation=28, fontsize=14)

# save figure
if args.save_fig or args.save_data: 
    out_dir=Path(args.out_dir)
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True)
if args.save_fig:
    fig.savefig(out_dir.joinpath(args.fig_name), dpi=300)

# save data
if args.save_data:
    data = pd.DataFrame(np.array((av_tops, sw_tops, sw_sds)),\
        columns = labels, index=['av_tops','sw_tops','sw_sds'])
    data.to_csv(out_dir.joinpath(args.data_name))
plt.show()
