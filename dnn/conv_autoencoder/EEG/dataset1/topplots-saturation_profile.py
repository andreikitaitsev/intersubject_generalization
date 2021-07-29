#! /bin/env/python3

import numpy as np
import joblib
import matplotlib.pyplot as plt
import argparse
import joblib
import pandas as pd
from pathlib import Path
from analysis_utils_conv_autoencoder import create_saturation_profile_data, plot_saturation_profile

parser= argparse.ArgumentParser()
parser.add_argument('-save_fig', action='store_true', default=False, help='Flag, save fig.')
parser.add_argument('-save_data', action='store_true', default=False, help='Flag, save data.')
parser.add_argument('-nsplits', type=str, default='100', help='N splits of training data.'
'Default=100.')
parser.add_argument('-steps', type=str, nargs='+', default=['5','10','20','40',\
'80','100'], help='Number of steps in incremental training data.')
parser.add_argument('-nshuffles',type=int, default=10, help='Default=10')
args=parser.parse_args()

srate='50hz'
inp_base=Path('/scratch/akitaitsev/intersubject_generalization/dnn/conv_autoencoder/conv_autoencoder_raw/EEG/dataset1/',\
    srate, ('incr-'+str(args.nshuffles)+'shuffle-'+str(args.nsplits)+'splits'))
av_fname='generic_decoding_results_average.pkl'
sw_fname='generic_decoding_results_subjectwise.pkl'

out_path=Path('/scratch/akitaitsev/intersubject_generalization/results/dnn/conv_autoencoder/saturation_profile/')

if not out_path.is_dir():
    out_path.mkdir(parents=True)

# get data
enc_av, enc_sw, dec_av, dec_sw = create_saturation_profile_data(inp_base, \
    args.nshuffles, args.steps)

# plot 
fig1, ax1 = plot_saturation_profile(enc_av, enc_sw, dec_av, dec_sw, layer='encoder',\
    labels=args.steps, fontsize=15)
fig2, ax2 = plot_saturation_profile(enc_av, enc_sw, dec_av, dec_sw, layer='decoder',\
    labels=args.steps, fontsize=15)

# save
base_dir = out_path.joinpath(('saturation_profile_'+str(args.nshuffles)+\
    'shuffles_'+str(args.nsplits)+'splits'))

if args.save_fig:
    fig1.savefig(Path(str(base_dir)+'_ENCODER.png'), dpi=300)
    fig2.savefig(Path(str(base_dir)+'_DECODER.png'), dpi=300)

if args.save_data:
    data = pd.DataFrame(np.array([enc_av['mean'], enc_av['sd'], enc_sw['mean'], \
        enc_sw['sd']]), index=['mean_av','sds_av','mean_sw','sds_sw'],\
        columns=[el + '%' for el in args.steps])
    data.to_csv(Path(str(base_dir)+'ENCODER.csv'))
    data = pd.DataFrame(np.array([dec_av['mean'], dec_av['sd'], dec_sw['mean'], \
        dec_sw['sd']]), index=['mean_av','sds_av','mean_sw','sds_sw'],\
        columns=[el + '%' for el in args.steps])
    data.to_csv(Path(str(base_dir)+'DECODER.csv'))
plt.show()
