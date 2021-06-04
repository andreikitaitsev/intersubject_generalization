#! /bin/env/python3

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from create_topplots import topplot_av_sw


'''Create topplots for control, mvica, permica with pca 200 for 
50hz, 100hz, 200hz.'''

import argparse 
parser = argparse.ArgumentParser(description='Create topplots for control (pca200), raw, '
'multiviewica and permica generic decoding results for 50, 100 and 200 hz.')
parser.add_argument('-top', type=int, default=1, help="Default=1.")
parser.add_argument('-out','--output_dir', type=str, 
default='/scratch/akitaitsev/intersubject_generalization/linear/dataset2',
help='Directory to save topplot pictures. '
'Default=/scratch/akitaitsev/intersubject_generalization/linear/dataset2/generic_decoding/results/')



av_fname = 'generic_decoding_results_average.pkl'
sw_fname = 'generic_decoding_results_subjectwise.pkl'
base_dir = Path('/scratch/akitaitsev/intersubject_generalization/linear/dataset2/generic_decoding/')
labels = ('av_raw', 'av_pca', 'av_mvica', 'av_permica', \
    'sw_raw', 'sw_pca', 'sw_mvica', 'sw_permica')

# time windows
tw50 = '13-40'
tw100 = '26-80'
tw200 = '52-160'

# 50 hz 
fls50 = (Path('raw/50hz/time_window').joinpath(tw50).joinpath('av_reps'), \
    Path('control/pca_200/50hz/time_window').joinpath(tw50).joinpath('av_reps'),\
    Path('multiviewica/pca_200/50hz/time_window').joinpath(tw50).joinpath('av_reps'), \
    Path('permica/pca_200/50hz/time_window').joinpath(tw50).joinpath('av_reps'))

fpaths50 = []
    for fl in fls50:
        fpaths50.append(base_dir.joinpath(fl))

fig50, ax50 = topplot_av_sw(av_fname, sw_fname, fpaths50, top=args.top, labels=labels, title=\
    ('Top '+str(args.top)+' generic decoding for time window 13-40, 50hz'))

# 100 hz 
fls100 = (Path('raw/100hz/time_window').joinpath(tw100).joinpath('av_reps'), \
    Path('control/pca_200/100hz/time_window').joinpath(tw100).joinpath('av_reps'),\
    Path('multiviewica/pca_200/100hz/time_window').joinpath(tw100).joinpath('av_reps'), \
    Path('permica/pca_200/100hz/time_window').joinpath(tw100).joinpath('av_reps'))

fpaths100 = []
    for fl in fls100:
        fpaths100.append(base_dir.joinpath(fl))

fig100, ax100 = topplot_av_sw(av_fname, sw_fname, fpaths100, top=args.top, labels=labels, title=\
    ('Top '+str(args.top)+' generic decoding for time window 26-80, 100hz'))

# 200 hz 
fls200 = (Path('raw/200hz/time_window').joinpath(tw200).joinpath('av_reps'), \
    Path('control/pca_200/200hz/time_window').joinpath(tw200).joinpath('av_reps'),\
    Path('multiviewica/pca_200/200hz/time_window').joinpath(tw200).joinpath('av_reps'), \
    Path('permica/pca_200/200hz/time_window').joinpath(tw200).joinpath('av_reps'))

fpaths200 = []
    for fl in fls200:
        fpaths200.append(base_dir.joinpath(fl))

fig200, ax200 = topplot_av_sw(av_fname, sw_fname, fpaths200, top=args.top, labels=labels, title=\
    ('Top '+str(args.top)+' generic decoding for time window 52-160, 200hz'))

fignames=( 'top_'+str(args.top)+'_time_window13-40_50hz', 'top_'+str(args.top)+'_time_window26-80_100hz',\
    'top_'+str(args.top)+'_time_window52-160_200hz')
figs=(fig50, fig100, fig200)
for num, fig in enumerate(figs):
    fig.savefig(Path(args.output_dir).joinpath(fignames[num]))
