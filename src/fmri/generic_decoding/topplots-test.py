#! /bin/env/python3

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from create_topplots import topplot_av_sw


'''Create topplots for test set control'''

import argparse 
parser = argparse.ArgumentParser(description='Create topplots for test control (pca200)')
parser.add_argument('-top', type=int, default=1, help="Default=1.")
parser.add_argument('-region', type=str, default='WB', help="Default=WB.")
parser.add_argument('-layer', type=str, default=None, help='Only use with layer-wise regression.') 
parser.add_argument('-inp','--input_dir', type=str, default=
'/scratch/akitaitsev/fMRI_Algonautus/generic_decoding/test/full_track/multiviewica/PCA/50/',
help="Default="
'/scratch/akitaitsev/fMRI_Algonautus/generic_decoding/test/full_track/multiviewica/PCA/50/')
parser.add_argument('-out','--output_dir', type=str, 
default='/scratch/akitaitsev/fMRI_Algonautus/results/test/',
help='Directory to save topplot pictures. '
'Default=/scratch/akitaitsev/fMRI_Algonautus/results/test/')
parser.add_argument('-save_fig', action='store_true', default=False)
args=parser.parse_args()


if not args.layer==None:
    av_fname = args.layer+'_'+str(args.region)+'generic_decoding_results_av.pkl'
    sw_fname = args.layer+'_'+str(args.region)+'generic_decoding_results_sw.pkl'
elif args.layer==None:
    av_fname = str(args.region)+'_generic_decoding_results_av.pkl'
    sw_fname = str(args.region)+'_generic_decoding_results_sw.pkl'
base_dir = [Path(args.input_dir)]
labels = ('av', 'sw')

fig, axs = topplot_av_sw(av_fname, sw_fname, base_dir, top=args.top, labels=labels, title=\
    ('Top '+str(args.top)+' generic decoding accuracy.'))

out_dir = Path(args.output_dir) 
figname='Top_'+str(args.top)+'_generic_decoding_accuracy.png'
if not out_dir.is_dir():
    out_dir.mkdir(parents=True)
plt.show()
if args.save_fig:
    fig.savefig(out_dir.joinpath(figname))
