import matplotlib.pyplot as plt
import numpy as np
import joblib
import matplotlib as mpl
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-res','--res_filepath', type=str, help='Filepath to generic decoding results to be plotted.')
parser.add_argument('-cor', '--cor_filepath', type=str, help='Filepath to generic decoding correlations matrices.')
parser.add_argument('-out_dir', type=str, help='Directory to save plots.')
args = parser.parse_args()

res = joblib.load(args.res_filepath)
cor = joblib.load(args.cor_filepath)
res = np.array(res)
cor = np.array(cor)

if 'subjectwise' in args.res_filepath:
    assert np.ndim(res) == 2
    d_type = 'subjectwise'
    # plot correlation matrices
    vmin = np.amin(cor)
    vmax = np.amax(cor)
    fig1, axes = plt.subplots(4,2, sharex=True, sharey=True, figsize=(16,9)) #7 subjs
    for subj, ax in zip(range(7), axes.flat):
        im = ax.imshow(cor[subj], origin='lower',aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title('Subject '+str(subj))
    cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    cbar=plt.colorbar(im, cax=cax, **kw)
    cbar.set_label('Correlation')
    
    # generic decoding bar plots
    fig2, axes =   plt.subplots(4,2, figsize=(16,9))
    for subj, ax in zip(range(7), axes.flat):
        res_it = res[subj,:]
        unique = set(res_it)
        n_el = len(res_it)
        unique_counter = [ np.sum([res_it[num] == un_el for num in range(len(res_it))])/n_el*100 for un_el in unique]
        plot = ax.bar(list(unique), unique_counter)
        ax.set_xlabel('Position of true image among n best correlated images.')
        ax.set_ylabel('Percent ratio.')
        ax.set_title('Subject '+str(subj)) 
    fig2.suptitle('Generic decoding results for subjectwise shared space.')

elif 'average' in args.res_filepath:
    assert np.ndim(res) == 1
    d_type='average'
    # plot correlation matrices
    vmin = np.amin(cor)
    vmax = np.amax(cor)
    fig1, ax = plt.subplots(figsize=(16,9)) 
    im = ax.imshow(cor, origin='lower',aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title('Average shared space correlation matrix')
    cbar = fig1.colorbar(im, ax=ax)
    cbar.set_label('Correlation')

    # generic decoding hist plots
    unique = set(res)
    n_el = len(res)
    unique_counter = [ np.sum([res[num] == un_el for num in range(len(res))])/n_el*100 for un_el in unique]
    fig2, ax = plt.subplots(figsize=(16,9))
    ax.bar(list(unique), unique_counter)
    ax.set_xlabel('Position of true image among n best correlated images.')
    ax.set_ylabel('Percent ratio')
    fig2.suptitle('Generic decoding results for shared space averaged between subjects')

# save figures to output dir
path = Path(args.out_dir)
if not path.is_dir():
    path.mkdir(parents=True)
fig1.savefig(path.joinpath(('correlation_matrices_'+d_type+'.png')), dpi=300)
fig2.savefig(path.joinpath(('percent_ratios_'+d_type+'.png')), dpi=300)

