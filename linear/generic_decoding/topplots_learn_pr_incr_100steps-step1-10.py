#! /bin/env/python3

import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from create_topplots import *

parser= argparse.ArgumentParser()
parser.add_argument('-top', type=int, default=1)
parser.add_argument('-methods', type=str, nargs='+', default=None, help='Default = '
'multiviewica')
parser.add_argument('-preprs',type=str, nargs='+', default=["pca"])
parser.add_argument('-n_comps', type=str, nargs='+', default=["200"])
parser.add_argument('-steps', type=str, nargs='+', default=['0','1','2','3',\
'4','5','6','7','8','9'], help='Number of steps in incremental training data.')
parser.add_argument('-save_fig', action='store_true', default=False, help='Flag, save figs')
args=parser.parse_args()

# change if needed
if args.methods == None:
    args.methods= ["multiviewica"]
inp_base=Path('/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/learn_projections_incrementally/')
title='Top '+str(args.top)+' generic decoding results for mvica with pca '+'_'.join(args.n_comps)


filepaths=[]
av_labels=[]
sw_labels=[]
percents=[]
av_fname='generic_decoding_results_average.pkl'
sw_fname='generic_decoding_results_subjectwise.pkl'

for meth in args.methods:
    for prepr in args.preprs:
        for n_comp in args.n_comps:
            for num, step in enumerate(args.steps):
                filepaths.append(inp_base.joinpath(meth, prepr, n_comp, '50hz','time_window13-40',\
                    '1-10percent', ('step_' +  str(step))))
                percents.append( '{:.0f}'.format((num+1)))
                av_labels.append('step '+ step)
                sw_labels.append('step '+ step)

labels=av_labels+sw_labels
percents=percents+percents
fig, ax = topplot_av_sw(av_fname, sw_fname, filepaths, top=args.top, labels=percents, title=title)
if not args.save_fig:
    plt.show()
if args.save_fig:
    out_path=Path('/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/plots/learn_projections_incrementally/50hz/1-10percent/')
    if not out_path.is_dir():
        out_path.mkdir(parents=True)
    fig.savefig(out_path.joinpath(('top'+str(args.top)+'_'+'_'.join(args.preprs)+'_'+\
        '_'.join(args.n_comps)+'.png')), dpi=300)
