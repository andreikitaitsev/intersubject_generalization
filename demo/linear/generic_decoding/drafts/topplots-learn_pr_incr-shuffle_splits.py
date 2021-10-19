#! /bin/env/python3

import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from create_topplots import *

parser= argparse.ArgumentParser()
parser.add_argument('-top', type=int, default=1)
parser.add_argument('-nsplits', type=str, default='10', help='N splits of training data.'
'Default=10')
parser.add_argument('-methods', type=str, nargs='+', default=None, help='Default = '
'multiviewica')
parser.add_argument('-preprs',type=str, nargs='+', default=["pca"])
parser.add_argument('-n_comps', type=str, nargs='+', default=["200"])
parser.add_argument('-nshuffles',type=int, default=10)
parser.add_argument('-steps', type=str, nargs='+', default=['0','1','2','3',\
'4','5','6','7','8','9'], help='Number of steps in incremental training data.')
parser.add_argument('-save_fig', action='store_true', default=False, help='Flag, save figs')
args=parser.parse_args()

# change if needed
if args.methods == None:
    args.methods= ["multiviewica"]
inp_base=Path('/scratch/akitaitsev/intersubject_generalization/linear/dataset1/generic_decoding/learn_projections_incrementally/',\
    'shuffle_splits/', (args.nsplits+'splits'))
title='Top '+str(args.top)+' generic decoding results for mvica with pca '+'_'.join(args.n_comps)

av_fname='generic_decoding_results_average.pkl'
sw_fname='generic_decoding_results_subjectwise.pkl'

legend=[]
legend_av=[]
legend_sw=[]

# plotting stuff
linestyles = ['solid','dotted']
colors=[ 'r', 'g', 'b']
xposerr=np.linspace(1, len(args.steps), len(args.steps), endpoint=True, dtype=int)
labels=np.linspace(1, len(args.steps), len(args.steps), endpoint=True, dtype=int)/\
    (int(args.nsplits)/100)
labels = [ '{:.0f}'.format(el) for el in labels]
xposerr_list=[0.1,0,-0.1]

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,9))
for meth in args.methods:
    for prepr in args.preprs:
        for n, n_comp in enumerate(args.n_comps):
            tops_av = []
            tops_sw=[]
            sds_av=[]
            sds_sw=[]
            for num, step in enumerate(args.steps): 
                fpaths_av_it=[]
                fpaths_sw_it=[]
                for shuffle in np.linspace(0, args.nshuffles, args.nshuffles, endpoint=False,dtype=int):
                    fpaths_av_it.append(inp_base.joinpath(meth, prepr, n_comp,'50hz','time_window13-40',\
                    ('shuffle_'+str(shuffle)), ('step_'+str(step)), av_fname))
                    fpaths_sw_it.append(inp_base.joinpath(meth, prepr, n_comp,'50hz','time_window13-40',\
                    ('shuffle_'+str(shuffle)), ('step_'+str(step)), sw_fname))
                # gen dec results for every shuffle in step N
                tops_av_it, _ = res2top(fpaths_av_it, args.top)
                tops_sw_it, sds_it =res2top(fpaths_sw_it, args.top)
                stack=lambda x: np.stack(x, axis=0)
                
                ## mean and sd over shuffles for step N
                tops_av.append(np.mean(stack(tops_av_it), axis=0))
                sds_av.append(np.std(stack(tops_av_it), axis=0))
                tops_sw.append(np.mean(stack(tops_sw_it), axis=0))
                sds_sw.append(np.mean(stack(sds_it)))
            
            # plot results
            legend_av.append('av_'+meth+'_'+prepr+'_'+n_comp)
            legend_sw.append('sw_'+meth+'_'+prepr+'_'+n_comp)
            title = 'Average generic decoding top ' +str(args.top) +' results and SDs over '+\
                str(args.nshuffles) + ' random shffles.'
            
            # plot data without errorbars
            fig, ax1 = topplot(tops_av, sds_av, labels= labels, \
                color=colors[n], linestyle=linestyles[0],\
                fig=fig, ax=ax1, graph_type='line', capsize=5,\
                xpos=xposerr-xposerr_list[n])
            fig, ax2 = topplot(tops_sw, sds_sw, labels= labels, \
                color=colors[n], linestyle=linestyles[1],\
                fig=fig, ax=ax2, graph_type='line', capsize=5,\
                xpos=xposerr-xposerr_list[n]) 

fig.suptitle(title)
ax1.legend(legend_av)
ax1.set_title('Average')
ax2.legend(legend_sw)
ax2.set_title('Subjectwise')

if args.save_fig:
    out_path=Path('/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/plots/',\
        'learn_projections_incrementally/shuffle_splits/')
    if not out_path.is_dir():
        out_path.mkdir(parents=True)
    fig.savefig(out_path.joinpath(('top'+str(args.top)+'_'.join(args.methods)+'_'.join(args.preprs)+\
        '_'.join(args.n_comps)+'_'+str(args.nsplits)+'splits'+'.png')), dpi=300)
if not args.save_fig:
    plt.show()
