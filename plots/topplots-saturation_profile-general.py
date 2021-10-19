#! /bin/bash/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from analysis_utils_linear import average_shuffles, saturation_topplot
from analysis_utils_dnn import create_saturation_profile_data

steps=['5', '10','20','40','80','100']
top=1

### Load data

# MVICA 
inp_mvica=Path('/scratch/akitaitsev/intersubject_generalization/linear/dataset1/generic_decoding/learn_projections_incrementally/',\
    'shuffle_splits/','100splits','100shuffles','multiviewica','pca','200','50hz','time_window13-40')
av_fname='generic_decoding_results_average.pkl'
sw_fname='generic_decoding_results_subjectwise.pkl'
nshuffles_mvica = 100

mvica_mean, mvica_sd, _, __ = average_shuffles(inp_mvica, av_fname, sw_fname, steps, nshuffles_mvica, top)

# CONV_AUTOENCODER
nshuffles_CAE=10
inp_CAE = Path('/scratch/akitaitsev/intersubject_generalization/dnn/conv_autoencoder/conv_autoencoder_raw/EEG/dataset1/',\
    '50hz', ('incr-'+str(nshuffles_CAE)+'shuffle-100splits'))
CAE, _, __, ___ = create_saturation_profile_data(inp_CAE, nshuffles_CAE, steps, net='conv_autoencoder')
CAE_mean=CAE['mean']
CAE_sd=CAE['sd']

# PERCEIVER
nshuffles_perc = 10
inp_perc = Path('/scratch/akitaitsev/intersubject_generalization/dnn/perceiver/EEG/dataset1/leftthomas/',\
    'projection_head', '50hz', ('incr-'+str(nshuffles_perc)+'shuffle-100splits'))
perc, _, __, ___ = create_saturation_profile_data(inp_perc, nshuffles_perc, steps, net='perceiver')
perc_mean = perc['mean']
perc_sd=perc['sd']


# plot saturation profiles
fig, ax = plt.subplots(figsize=(12,8))

xpos2 = np.linspace(0, len(steps), len(steps), endpoint=False, dtype=int)
xpos1 = xpos2-0.01
xpos3 = xpos2+0.01

fig, ax = saturation_topplot(mvica_mean, mvica_sd, fig=fig, ax=ax, xtick_labels = steps, color = 'C1',\
    graph_type='line', xpos=xpos1)
fig, ax = saturation_topplot(perc_mean, perc_sd, fig=fig, ax=ax, xtick_labels = steps, color = 'C2',\
    graph_type='line', xpos=xpos2)
fig, ax = saturation_topplot(CAE_mean, CAE_sd, fig=fig, ax=ax, xtick_labels = steps, color = 'C3',\
    graph_type='line', xpos=xpos3)
ax.legend(['MultiViewICA', 'Perceiver', 'CAE'])

parser= argparse.ArgumentParser()
parser.add_argument('-save_fig', action='store_true', default=False, help='Flag, save fig.')
args=parser.parse_args()
out_dir=Path('/scratch/akitaitsev/intersubject_generalization/results/general/')
if args.save_fig:
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True)
    fig.savefig(out_dir.joinpath('saturation_profile_encoder_average.png'), dpi=300)

plt.show()
