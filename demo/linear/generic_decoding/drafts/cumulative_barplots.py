#! /bin/env/python
'''Script to create barplots of 1)raw_data pca200 2)mvica pca200
subjectwise and 3)mvica p200 average. ''' 

import matplotlib.pyplot as plt
import numpy as np
import joblib
import matplotlib as mpl
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-ctrl','--control_filepath', type=str, help='Filepath to generic decoding results on '
'control (raw data of the same dimensionality as mvica data).')
parser.add_argument('-sw', '--subjectwise_filepath', type=str, help='Filepath to generic decoding results on '
'subjectwise data.')
parser.add_argument('-av', '--average_filepath', type=str, help='Filepath to generic decoding results .'
'on averaged data.')
parser.add_argument('-out','--output_dir', type=str, help='Directory to save plots.')
args = parser.parse_args()

ctrl = joblib.load(args.control_filepath)
sw = joblib.load(args.subjectwise_filepath)
av = joblib.load(args.average_filepath)

# convert results to histograms
res2hist = lambda x: [ np.sum([x[n] == u for n in range(len(x))])*100/len(x) for u in set(x)]
ctrl_hist = [res2hist(i) for i in ctrl]
sw_hist = [res2hist(subj) for subj in sw]
av_hist = res2hist(av)

fig, axes = plt.subplots(4,2, figsize=(16,9))
for subj, ax in zip(range(7), axes.flat):
    labels_sw=[el-0.3 for el in list(set(sw[subj]))]
    labels_av=list(set(av))
    labels_ctrl = [el +0.3 for el in list(set(ctrl[subj]))]
    plot_av = ax.bar(labels_av, av_hist, 0.3)
    plot_sw = ax.bar(labels_sw, sw_hist[subj],0.3) # subjectwise hist
    plot_ctrl = ax.bar(labels_ctrl, ctrl_hist[subj],0.3)
    #max_im_num = int(max(max(labels_sw),max(labels_av),max(labels_ind)))
    max_im_num=10
    ax.set_xlim(0,max_im_num)
    ax.set_ylim(0,100)
    ax.set_xticks(list(map(int,np.linspace(1,max_im_num,max_im_num, endpoint=True).tolist())) ) 
    ax.set_xlabel('Position of true image among n best correlated images')
    ax.set_ylabel('Percent ratio')
    ax.set_title('Subj '+str(subj))
plt.subplots_adjust(hspace=0.6, wspace=0.5)
fig.legend(['mvica average','mvica subjectwise', 'control (pca200)'])
fig.suptitle('Generic decoding results.')

# save figure and histogram dictionary
parts=Path(args.control_filepath).parts
if 'time_window0-40' in parts:
    time='time_window0-40'
elif 'time_window13-40' in parts:
    time='time_window13-40'
hist_dict = {'control':ctrl_hist, 'subjectwise':sw_hist,'average':av_hist}
#joblib.dump(hist_dict, Path(args.output_dir).joinpath('cumulative_decoding_results_histograms.pkl'))
fig.savefig(Path(args.output_dir).joinpath(('cumulative_decoding_results_'+time+'.png')),dpi=300)
