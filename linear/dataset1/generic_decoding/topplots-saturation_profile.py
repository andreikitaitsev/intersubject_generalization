#! /bin/env/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from analysis_utils import average_shuffles, plot_saturation_profile


parser= argparse.ArgumentParser()
parser.add_argument('-save_fig', action='store_true', default=False, help='Flag, save fig.')
parser.add_argument('-save_data', action='store_true', default=False, help='Flag, save data.')
parser.add_argument('-nsplits', type=str, default='100', help='N splits of training data.'
'Default=100.')
parser.add_argument('-steps', type=str, nargs='+', default=['5','10','20','40',\
'80','100'], help='Number of steps in incremental training data.')
parser.add_argument('-nshuffles',type=int, default=100)
parser.add_argument('-top', type=int, default=1)
parser.add_argument('-methods', type=str, nargs='+', default=['multiviewica'], help='Default = '
'multiviewica')
parser.add_argument('-preprs',type=str, nargs='+', default=["pca"])
parser.add_argument('-n_comps', type=str, nargs='+', default=["200"])
args=parser.parse_args()

inp_base=Path('/scratch/akitaitsev/intersubject_generalization/linear/dataset1/generic_decoding/learn_projections_incrementally/',\
    'shuffle_splits/', (args.nsplits+'splits'), (str(args.nshuffles)+'shuffles'))
av_fname='generic_decoding_results_average.pkl'
sw_fname='generic_decoding_results_subjectwise.pkl'

fpaths = []
for meth in args.methods:
    for prepr in args.preprs:
        for n_comp in args.n_comps:
            fpaths.append(inp_base.joinpath(meth, prepr, n_comp, '50hz', 'time_window13-40')) 

out_path=Path('/scratch/akitaitsev/intersubject_generalization/results/linear/saturation_profile/')
if not out_path.is_dir():
    out_path.mkdir(parents=True)

# for methods for preprocessors for n_components
for fpath in fpaths:
    tops_av, sds_av, tops_sw, sds_sw = average_shuffles(fpath, av_fname, \
        sw_fname, args.steps, args.nshuffles, args.top)
    
    fig, ax = plot_saturation_profile(tops_av, sds_av, tops_sw, sds_sw, xtick_labels=args.steps,\
        top=1, fontsize=15)

    base_name = out_path.joinpath(('top_'+str(args.top)+'_'.join(args.methods)+'_'.join(args.preprs)+\
        '_'.join(args.n_comps)+'_'+str(args.nsplits)+'splits'))
    
    if args.save_fig:
        fig.savefig(Path(str(base_name)+'.png'))

    if args.save_data:
        data = pd.DataFrame(np.array([tops_av, sds_av, tops_sw, sds_sw]), \
            index=['tops_av','sds_av','tops_sw', 'sds_sw'],\
            columns=[el +'%' for el in args.steps])
        data.to_csv(Path(str(base_name)+'.csv'))
    plt.show()

