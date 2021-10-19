#! /bin/env/python3

import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

parser= argparse.ArgumentParser()
parser.add_argument('-top', type=int, default=1)
parser.add_argument('-seeds', nargs='+', type=list, default=[0,1,2,3,4,5,6,7,8,9], help='Sequene of used seeds.')
parser.add_argument('-steps', type=str, nargs='+', default=['5','10','20','40',\
'80','100'], help='Number of steps in incremental training data.')
parser.add_argument('-save_fig', action='store_true', default=False, help='Flag, save figs.')
parser.add_argument('-save_fig_dir', default=None, type=str, help='Dir to save fig.')

args=parser.parse_args()

inp_base=Path('/scratch/akitaitsev/intersubject_generalization/dnn/perceiver/EEG/dataset1/leftthomas/projection_head/50hz/incr-1shuffle-100splits/')

accs_enc_av=[]
accs_ph_av=[]
accs_enc_sw=[]
accs_ph_sw=[]

for step in args.steps:
    accs_enc_av_it=[]
    accs_ph_av_it=[]
    accs_enc_sw_it=[]
    accs_ph_sw_it=[]
    for seed in args.seeds:
        path_it = inp_base.joinpath(('seed'+str(seed)), ('step'+str(step)), 'test_accuracies.pkl' )
        accs_it = joblib.load(path_it)
        # select maximal value from all the assessments
        accs_enc_av_it.append(max(accs_it["encoder"]["average"]))
        accs_ph_av_it.append(max(accs_it["projection_head"]["average"]))
        accs_enc_sw_it.append(max(accs_it["encoder"]["subjectwise"]["mean"]))
        accs_ph_sw_it.append(max(accs_it["projection_head"]["subjectwise"]["mean"]))
    accs_enc_av.append(accs_enc_av_it)
    accs_ph_av.append(accs_ph_av_it)
    accs_enc_sw.append(accs_enc_sw_it)
    accs_ph_sw.append(accs_ph_sw_it)

# average over shuffles
accs_enc_av=np.array(accs_enc_av)
accs_ph_av=np.array(accs_ph_av) 
accs_enc_sw=np.array(accs_enc_sw)
accs_ph_sw=np.array(accs_ph_sw)

mean_enc_av=np.mean(accs_enc_av, axis=0)
mean_ph_av=np.mean(accs_ph_av, axis=0)
mean_enc_sw=np.mean(accs_enc_sw, axis=0)
mean_ph_sw=np.mean(accs_ph_sw, axis=0)

sd_enc_av=np.std(accs_enc_av, axis=0)
sd_ph_av=np.std(accs_ph_av, axis=0)
sd_enc_sw=np.std(accs_enc_sw, axis=0)
sd_ph_sw=np.std(accs_ph_sw, axis=0)


fig, ax=plt.subplots(2, figsize=(16,9), sharex=True, sharey=True)
ax[0].errorbar(args.steps, mean_enc_av, yerr=sd_enc_av)
ax[0].errorbar(args.steps, mean_enc_sw, yerr=sd_enc_sw)
ax[0].set_ylim(0,100)
ax[0].set_xlabel('ratio of training images used,%')  
ax[0].set_title('encoder')
ax[0].legend(['average','subject-wise'])
ax[1].errorbar(args.steps, mean_ph_av, yerr=sd_ph_av)
ax[1].errorbar(args.steps, mean_ph_sw, yerr=sd_ph_sw)
ax[1].set_ylim(0,100)
ax[1].set_title('projection head')
ax[1].legend(['average','subject-wise'])

if args.save_fig:
    fig.savefig(Path(args.save_fig_path), dpi=300)

plt.show()
