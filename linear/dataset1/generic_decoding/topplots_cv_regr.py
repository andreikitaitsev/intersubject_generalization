#! /bin/env/python3
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from analysis_utils import res2top

def topplot(tops, errors, labels=None, xpos=None, title=None):
    
    if xpos==None: 
        xpos=np.arange(0, len(tops), 1, dtype=int)
    
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_axes([0.05,0.05,0.9, 0.88])
    ax.bar(xpos, tops, yerr=errors, color='b', align='center', capsize=10)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_ylim([0,100])
    ax.set_ylabel('Percent ratio')
    fig.suptitle(title)
    return fig, ax

if __name__ =='__main__':
    import argparse
    parser= argparse.ArgumentParser()
    parser.add_argument('-top', type=int, default=1)
    parser.add_argument('-methods', type=str, nargs='+', default=None, help='Default = '
    'multiviewica, groupica, permica')
    parser.add_argument('-preprs',type=str, nargs='+', default=["pca"])
    parser.add_argument('-n_comps', type=str, nargs='+', default=["200"])
    parser.add_argument('-save_fig', action='store_true', default=False, help='Flag, save figs')
    args=parser.parse_args()

    # change if needed
    if args.methods == None:
        args.methods= ["multiviewica","groupica", "permica"]
    n_comps=['50', '200', '400']
    inp_base=Path('/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/generic_decoding/cv_regr/')
    title='Top '+str(args.top)+' generic decoding accuracy'
    
    
    filepaths=[]
    labels=[]
    for meth in args.methods:
        for prepr in args.preprs:
            for n_comp in args.n_comps:
                filepaths.append(inp_base.joinpath(meth, prepr, n_comp, '50hz','time_window13-40', \
                    'generic_decoding_results.pkl'))
                labels.append((meth+'_'+prepr+'_'+n_comp))
    tops, sds = res2top(filepaths, args.top)
    fig, ax = topplot(tops, sds, labels=labels, title=title)
    if not args.save_fig:
        plt.show()
    if args.save_fig:
        out_path=Path('/scratch/akitaitsev/intersubject_generalizeation/results/linear/cv_regr/')
        if not out_path.is_dir():
            out_path.mkdir(parents=True)
        fig.savefig(out_path.joinpath(('top'+str(args.top)+'_'+'_'.join(args.preprs)+'_'+\
            '_'.join(args.n_comps)+'.png')), dpi=300)
