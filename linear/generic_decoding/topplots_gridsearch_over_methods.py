''' Plot top 1, top5 and top 10 graphs for gridsearch'''
#! /bin/env/python
from create_topplots import topplot_av_sw
from pathlib import Path
import matplotlib.pyplot as plt

## define data of interest

# define project dir and time window of interest
output_dir=Path('/scratch/akitaitsev/intersubject_generalizeation/linear/'\
    'generic_decoding/plots/gridsearch/over_methods/50hz/')
project_dir = Path('/scratch/akitaitsev/intersubject_generalizeation/linear')
time_window='time_window13-40'
av_filename='generic_decoding_results_average.pkl'
sw_filename='generic_decoding_results_subjectwise.pkl'
filepaths= []
labels=[]

# define control filepaths for contorl - raw data 
filepaths.append(project_dir.joinpath('generic_decoding','control','main','raw',\
    time_window))
labels.append('raw') # name on topplot_av_sw

# define control filepaths for control - pca200 
filepaths.append(project_dir.joinpath('generic_decoding','control','main','pca','200',\
    '50hz', time_window))
labels.append('pca')

# define methods filepaths
methods = ['multiviewica', 'groupica', 'permica']
method_aliases=['mvica','gica','pica']
preprocessors = ['pca', 'srm']
for method, alias in zip(methods, method_aliases):
    for preprocessor in preprocessors:
        filepaths.append(project_dir.joinpath('generic_decoding', method,\
            'main',preprocessor, '200', time_window))
        labels.append( (alias+'_'+preprocessor))

## Topplots
# create final labels
av_labels=[]
sw_labels=[]
for el in labels:
    av_labels.append(('av_'+el))
    sw_labels.append(('sw_'+el))
final_labels = av_labels+sw_labels

# top 1 plots
fig1, ax1 = topplot_av_sw(av_filename, sw_filename, filepaths, top=1, labels=final_labels,\
    title='Generic decoding top 1 results for time window 13 40')
# top 5 plots 
fig2, ax2 = topplot_av_sw(av_filename, sw_filename, filepaths, top=5, labels=final_labels,\
    title='Generic decoding top 5 results for time window 13 40')
# top 10 plots
fig3, ax3 = topplot_av_sw(av_filename, sw_filename, filepaths, top=10, labels=final_labels,\
    title='Generic decoding top 10 results for time window 13 40')

if not output_dir.is_dir():
    output_dir.mkdir(parents=True)

fig1.savefig(output_dir.joinpath('top1_time_window13-40.png'), dpi=300)
fig2.savefig(output_dir.joinpath('top5_time_window13-40.png'), dpi=300)
fig3.savefig(output_dir.joinpath('top10_time_window13-40.png'), dpi=300)
