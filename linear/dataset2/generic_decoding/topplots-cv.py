#! /bin/env/python3
'''Topplots for transfer learning experiment (train IGA on 10 and 100 % of
the training dataset and then predict shared space test set using 
leave-one-out CV.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import joblib
from pathlib import Path
from analysis_utils_linear import res2hist, hist2top, res2top


parser= argparse.ArgumentParser()
parser.add_argument('-save_fig', action='store_true', default=False, help='Flag, save fig.')
#parser.add_argument('-save_data', action='store_true', default=False, help='Flag, save data.')
parser.add_argument('-top', type=int, default=1)
parser.add_argument('-methods', type=str, nargs='+', default=['control10-cv0','control100-cv0',\
    'control10-cv1', 'control100-cv1', 'multiviewica', 'groupica','permica'],\
help='Default = control10-cv0, control100-cv0, control10-cv1, control100-cv1, multiviewica, groupica, permica')
parser.add_argument('-inp_dir', type=str, default='/scratch/akitaitsev/intersubject_generalization/linear/dataset2/generic_decoding/cv/',\
help='Defalut=/scratch/akitaitsev/intersubject_generalization/linear/dataset2/generic_decoding/cv/')
parser.add_argument('-out_dir', type=str, default='/scratch/akitaitsev/intersubject_generalization/results/linear/cv/',\
help='Default = /scratch/akitaitsev/intersubject_generalization/results/linear/cv/')
parser.add_argument('-fig_name', type=str, default='dataset2-cv-comparison.png',\
help='Default = dataset2-cv-comparison.png')
args=parser.parse_args()

# get file paths
fpaths = []
inp_base=Path(args.inp_dir)
fname = 'generic_decoding_results_.pkl'
for meth in args.methods:
    fpaths.append(inp_base.joinpath(meth, '100hz', 'time_window26-80', fname))

# get mean and sd (over CV splits) for every method
means=[]
sds=[]
for fpath in fpaths:
    fl = joblib.load(fpath)
    hist = res2hist(np.array(fl))
    mean, sd = hist2top(hist, args.top, return_sd=True)
    means.append(mean)
    sds.append(sd)

# plot results as a barplot
fig, ax = plt.subplots(figsize=(12,9))
x=np.linspace(0, len(args.methods), len(args.methods), endpoint=False, dtype=int)
ax.bar(x, means, yerr=sds, capsize=6, color='C0')
ax.set_xticks(x)
ax.set_xticklabels(args.methods)
ax.set_ylim([0,105])
ax.set_ylabel('Top1 geenric decoding accuracy, %')

# save data
out_dir=Path(args.out_dir)
if not out_dir.is_dir():
    out_dir.mkdir(parents=True)

if args.save_fig:
    fig.savefig(out_dir.joinpath(args.fig_name))

plt.show()
