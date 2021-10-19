#! /bin/env/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from analysis_utils_linear import average_shuffles, plot_saturation_profile_different_methods


parser= argparse.ArgumentParser()
parser.add_argument('-save_fig', action='store_true', default=False, help='Flag, save fig.')
parser.add_argument('-save_data', action='store_true', default=False, help='Flag, save data.')
parser.add_argument('-nsplits', type=str, default='100', help='N splits of training data.'
'Default=100.')
parser.add_argument('-steps', type=str, nargs='+', default=['10','20','30','40',\
'50', '60', '70', '80', '90','100'], help='Number of steps in incremental training data.')
parser.add_argument('-nshuffles',type=int, default=10)
parser.add_argument('-top', type=int, default=1)
parser.add_argument('-methods', type=str, nargs='+', default=['multiviewica', 'groupica','permica'],\
help='Default = multiviewica, groupica, permica')
parser.add_argument('-inp_dir', type=str, default='/scratch/akitaitsev/intersubject_generalization/linear/dataset2/generic_decoding/saturation_profile-different_methods/', help='Default=/scratch/akitaitsev/intersubject_generalization/linear/dataset2/generic_decoding/'\
'saturation_profile-diffrent_methods/')

args=parser.parse_args()

inp_base=Path(args.inp_dir)
av_fname='generic_decoding_results_average.pkl'
sw_fname='generic_decoding_results_subjectwise.pkl'

fpaths = []
for meth in args.methods:
    fpaths.append(inp_base.joinpath(meth, '50hz'))

out_path=Path('/scratch/akitaitsev/intersubject_generalization/results/linear/saturation_profile-different_methods/dataset2/')
if not out_path.is_dir():
    out_path.mkdir(parents=True)

# for methods 
fig, ax = plt.subplots(1,2, figsize=(12,9))
tops_av=[]
sds_av=[]
tops_sw=[]
sds_sw=[]
for fpath in fpaths:
    tops_av_, sds_av_, tops_sw_, sds_sw_ = average_shuffles(fpath, av_fname, \
        sw_fname, args.steps, args.nshuffles, args.top)
    tops_av.append(tops_av_)
    tops_sw.append(tops_sw_)
    sds_av.append(sds_av_)
    sds_sw.append(sds_sw_)

xpos_base=np.linspace(1,len(args.steps), len(args.steps), endpoint=True)
xpos=[]
xpos=[xpos_base-0.1, xpos_base, xpos_base+0.1]

fig, ax = plot_saturation_profile_different_methods(tops_av, sds_av, tops_sw, sds_sw, xtick_labels=args.steps,\
    xpos=xpos, top=1, fontsize=15, labels=args.methods)

base_name = out_path.joinpath(('top_'+str(args.top)+'_'.join(args.methods)+\
    '_'+str(args.nsplits)+'splits'))

if args.save_fig:
    fig.savefig(Path(str(base_name)+'.png'))

if args.save_data:
    ar=np.array([tops_av, sds_av, tops_sw, sds_sw])
    ar=ar.reshape(ar.shape[0]*ar.shape[1], ar.shape[2])
    idx=pd.MultiIndex.from_product([('tops_av','sds_av','tops_sw', 'sds_sw'), \
        tuple(args.methods)])
    data = pd.DataFrame(ar, index=idx, columns=[el +'%' for el in args.steps])
    data=data.stack().reset_index()
    data.set_axis(['dtype','method', 'step', 'value'], axis=1, inplace=True)
    data.to_csv(Path(str(base_name)+'.csv'))
plt.show()

