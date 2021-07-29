#! /bin/env/python
from create_topplots import res2top
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# filepaths
output_dir = Path('/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/'\
    'generic_decoding/plots/control/50hz/')

project_dir = Path('/scratch/akitaitsev/intersubject_generalization/linear/dataset1/')
time_window = 'time_window13-40'
av_filename = 'generic_decoding_results_average.pkl'
sw_filename = 'generic_decoding_results_subjectwise.pkl'

raw_av_path = [project_dir.joinpath('generic_decoding','control','main','raw',\
    time_window, av_filename)]
raw_sw_path = [project_dir.joinpath('generic_decoding','control','main','raw',\
    time_window, sw_filename)]

pca_av_path = [project_dir.joinpath('generic_decoding','control','main','pca','200','50hz',\
    time_window, av_filename)]
pca_sw_path = [project_dir.joinpath('generic_decoding','control','main','pca','200', '50hz',\
    time_window, sw_filename)]

# generic decoding results
top=1
raw_av, _ = res2top(raw_av_path, top=top)
raw_sw_mean, raw_sw_sd = res2top(raw_sw_path, top=top)
pca_av, _ = res2top(pca_av_path, top=top)
pca_sw_mean, pca_sw_sd = res2top(pca_sw_path, top=top)

av=[raw_av[0], pca_av[0]]
sw_mean=[raw_sw_mean[0], pca_sw_mean[0]]
sw_sd=[raw_sw_sd[0], pca_sw_sd[0]]

fig, axes = plt.subplots(1, 2, figsize=(16,9))
labels = ['raw', 'PCA']
x=np.linspace(0, len(av), len(av), endpoint=False, dtype=int)
axes[0].bar(x, av, color='C1', align='center', capsize=10)
axes[0].set_title('average')
axes[1].bar(x, sw_mean, yerr=sw_sd, color='C2', align='center', capsize=10)
axes[1].set_title('subject-wise')

for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0,100)
    ax.set_ylabel('Top 1 accuracy, %')

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-save_fig', action='store_true', default=False, help='Flag, whether to save '
    'figure. Default=False.')
    parser.add_argument('-save_fig_dir', type=str, 
    default='/scratch/akitaitsev/intersubject_generalization/linear/dataset1/plots/', 
    help='Directory to save figure. Default=/scratch/akitaitsev/intersubject_generalization/'
    'linear/dataset1/plots/.')
    parser.add_argument('-fig_name', type=str, 
    default='control.png', help='Figure name. Default=control.png')
    args = parser.parse_args()
    if args.save_fig: 
        out_dir=Path(args.save_fig_dir)
        if not out_dir.is_dir():
            outdir.makedirs(parents=True)
        fig.savefig(out_dir.joinpath(args.fig_name), dpi=300)
    plt.show()
