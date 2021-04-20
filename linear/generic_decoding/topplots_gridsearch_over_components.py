''' Plot top 1, top5 and top 10 graphs for gridsearch'''
#! /bin/env/python
from create_topplots import topplot
from pathlib import Path
import matplotlib.pyplot as plt

## define data of interest

# define project dir and time window of interest
output_dir=Path('/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/plots/gridsearch/over_components/50hz/')
project_dir = Path('/scratch/akitaitsev/intersubject_generalizeation/linear')
time_window='time_window13-40'
av_filename='generic_decoding_results_average.pkl'
sw_filename='generic_decoding_results_subjectwise.pkl'
filepaths= []
labels=[]

for preprocessor in ['srm','pca']:
    for n_comp in ['10','50','200','400']:
        labels.append(preprocessor+'_'+str(n_comp))
        filepaths.append(project_dir.joinpath('generic_decoding','multiviewica','main',\
        preprocessor, n_comp, '50hz', time_window))

## Topplots
# create final labels
av_labels=[]
sw_labels=[]
for el in labels:
    av_labels.append(('av_'+el))
    sw_labels.append(('sw_'+el))
final_labels = av_labels+sw_labels

# top 1 plots
fig1, ax1 = topplot(av_filename, sw_filename, filepaths, top=1, labels=final_labels,\
    title='Generic decoding top 1 results for multiviwica time window 13 40')
# top 5 plots 
fig2, ax2 = topplot(av_filename, sw_filename, filepaths, top=5, labels=final_labels,\
    title='Generic decoding top 5 results for multiviewica time window 13 40')
# top 10 plots
fig3, ax3 = topplot(av_filename, sw_filename, filepaths, top=10, labels=final_labels,\
    title='Generic decoding top 10 results for multiviewica time window 13 40')

if not output_dir.is_dir():
    output_dir.mkdir(parents=True)
fig1.savefig(output_dir.joinpath('top1.png'), dpi=300)
fig2.savefig(output_dir.joinpath('top5.png'), dpi=300)
fig3.savefig(output_dir.joinpath('top10.png'), dpi=300)
