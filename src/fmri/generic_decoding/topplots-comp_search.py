#! /bin/env/python3

'''Create topplots for different number of PCA for each ROI (excluding the Whole Brain).'''
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from analysis_utils_linear import res2top_single_file
from generic_decoding import scan_ncomp_from_dir
import argparse 

parser = argparse.ArgumentParser(description='Create topplots for different number of '
    'PCs for each ROI (including Whole Brain).''')
parser.add_argument('-top', type=int, default=1, help="Default=1.")
parser.add_argument('-regions', nargs='+', type=str, default=None, help="List of regions to use."
"Default = all (WB, EBA, FFA, LOC, PPA, STS, V1, V2, V3, V4)")
parser.add_argument('-comps', nargs='+', type=str, default=None, help='Number of components. '
'If None, auto_scan from folder provied.')
parser.add_argument('-inp', type=str, default=
'/scratch/akitaitsev/fMRI_Algonautus/generic_decoding/eigPCA/', help='Input directory. '
'Default=/scratch/akitaitsev/fMRI_Algonautus/generic_decoding/eigPCA/')
parser.add_argument('-out','--output_dir', type=str, 
    default='/scratch/akitaitsev/fMRI_Algonautus/results/',
    help='Directory to save topplot pictures. Default = '
    '/scratch/akitaitsev/fMRI_Algonautus/results/')
args=parser.parse_args()


av_fname = 'generic_decoding_results_av.pkl'
sw_fname = 'generic_decoding_results_sw.pkl'
base_dir = Path('/scratch/akitaitsev/fMRI_Algonautus/generic_decoding/comp_search/multiviewica/eig_PCA/')
if args.regions is None:
    regions=('WB', 'EBA', 'FFA', 'LOC', 'PPA', 'STS', 'V1', 'V2', 'V3', 'V4')
else:
    regions = args.regions

tops_av=[]
tops_sw_av=[]
tops_sw_sds=[]
#av
for region in regions:
    if args.comps == None:
        comps = scan_ncomp_from_dir(base_dir, [region])[region]
    else:
        comps = args.comps
    for comp in comps:
        base_dir_it=base_dir.joinpath(region, ('PCA'+comp))
        av_fname_it = region+av_fname
        file_av=joblib.load(base_dir_it.joinpath(av_fname_it))
        top1_av_it, _ = res2top_single_file(file_av, args.top)  
        tops_av.append(top1_av_it)

# sw
for region in regions:
    if args.comps == None:
        comps = scan_ncomp_from_dir(base_dir, [region])[region]
    else:
        comps = args.comps
    for comp in comps:
        base_dir_it=base_dir.joinpath(region, ('PCA'+comp))
        sw_fname_it = region+sw_fname
        file_sw=joblib.load(base_dir_it.joinpath(sw_fname_it))
        top1_sw_av_it, top1_sw_sd_it = res2top_single_file(file_sw, args.top)  
        tops_sw_av.append(top1_sw_sd_it)
        tops_sw_sds.append(top1_sw_sd_it)

def create_x(ncomp, nregions, gap=2):
    '''Create x axis for the barplot.'''
    createX=lambda ncomp, nregions, gap: [np.linspace(1,ncomp,ncomp,\
        endpoint=True)+ncomp*num+gap*num for num in range(nregions)]
    x = createX(ncomp, nregions, gap)
    if ncomp%2 == 0:
        idx = len(x[0])
        xticks =[np.mean(ar[idx], ar[idx+1]) for ar in x ]
    elif ncomp%2 !=0:
        idx = np.ceil(len(x[0])/2)
        xtikcs=[np.mean(ar[idx], ar[idx+1]) for ar in x ]
    return np.concatenate(x), np.array(xticks)

def plot_compare_pcs(x, xticks, xticklabels, tops_av, tops_sw, sds_sw):
    fig, axs = plt.subplots(1,2, figsize=(12,8))
    axs[0].bar(x, tops_av)
    axs[0].set_title('Average', fontsize=16)
    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels(xticklabels, fontsize=14)
    axs[1].errorbar(x, tops_sw, yerr=sds_sw, capsize=8)
    axs[1].set_title('Subjectwise', fontsize=16)
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(xticklabels, fontsize=14)
    return fig, axs

def createX(ncomp, nregions, gap):
    ar = np.linspace(0, ncomp*nregions, ncomp*nregions, endpoint=False)
    ars = np.array_split(ar, nregions)
    for n in range(len(ars)):
        ars[n] = ars[n] + gap*n
    return ars

def createX_ticks(ncomp, nregions, gap):
    n1 = np.floor(ncomp/2)
    ns=[]
    for n in range(nregions):
        ns.append(n1 + gap*n + ncomp*n)
    return ns


flatten_list = lambda x: np.concatenate(x).tolist() 
x=flatten_list(createX(3, 10, 2))
xticks=createX_ticks(3,10,2)

fig, ax = plot_compare_pcs(x, xticks, regions, tops_av, tops_sw_av, tops_sw_sds)
plt.show()

figname='Top'+str(args.top)+'comp_search.png'
out_dir = Path(args.output_dir) 
if not out_dir.is_dir():
    out_dir.mkdir(parents=True)
fig.savefig(out_dir.joinpath(figname))
