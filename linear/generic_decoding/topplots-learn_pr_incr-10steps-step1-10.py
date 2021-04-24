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
inp_base=Path('/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/learn_projections_incrementally/',\
    'shuffle_splits/10splits/multiviewica/pca/')
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
            tops_av = []
            tops_sw=[]
            sds_av=[]
            sds_sw=[]
            for num, step in enumerate(args.steps): 
                fpaths_av_it=[]
                fpath_sw_it=[]
                for shuffle in np.linspace(0, args.nshuffles, args.nshuffles, endpoint=False,dtype=int):
                    fpaths_av_it.append(inp_base.joinpath(str(args.n_comps),'50hz','time_window13-40','shuffle_0',\
                        ('step_'+str(step)), av_fname))
                    fpaths_sw_it.append(inp_base.joinpath(str(args.n_comps),'50hz','time_window13-40','shuffle_0',\
                        ('step_'+str(step)), sw_fname))
                # gen dec results for every shuffle in step N
                tops_av_it = res2top(fpaths_av_it, args.top)
                tops_sw_it, sds_it =res2top(fpaths_sw_it, args.top)
                stack=lambda x: np.stack(x, axis=0)
                
                ## results for step N
                tops_av.append(np.mean(stack(tops_av_it), axis=0))
                sds_av.append(np.std(stack(tops_av_it), axis=0))
                tops_sw.append(np.mean(stack(top_sw_it), axis=0))
                # sd over subjects averaged over shuffles for step N
                sds_sw.append(np.mean(np.concatenate(sds_it)))
            # plot results
            tops_av=np.concatenate(tops_av)
            sds_av=np.concatenate(sds_av)
            tops_sw=np.concatenate(tops_sw)
            sds_sw=np.concatenate(sds_sw)

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
