#! /bin/env/python3
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

def res2hist(fl):
    '''Convert generic decoding results into histograms)
    In the current project data structures, the output files are
    named the same and the paths differ.
    Inputs:
        fl - 2d or 1d np array of generic_decoding_result_average/subjectwise
    Outputs:
        hist - list of arrays of histogram or list of lists of
               arrays of histograms for different subjects
    
    '''
    hist=[]
    if np.ndim(fl) == 2:
        for s in range(fl.shape[0]):
            hist_tmp=np.histogram(fl[s], np.linspace(1,\
                max(set(fl[s]))+1, max(set(fl[s])), endpoint=False, dtype=int))
            hist.append( (hist_tmp[0]/sum(hist_tmp[0]))*100 )
    elif np.ndim(fl) ==1:
        hist_tmp = np.histogram(fl, np.linspace(1,max(set(fl))+1,\
            max(set(fl)), endpoint=False, dtype=int))
        hist.append((hist_tmp[0]/sum(hist_tmp[0]))*100)
    return hist

def hist2top(hist, top, return_sd=False):
    '''Returns the percent of images in top N best correlated images.
    Inputs:
        hist - list of arrays (possibly for different subjects),
               output of res2hist function
        top - int, position of the image (starting from 1!)
        return_sd - bool, whether to return sd over subjects. Default=False
    Returns:
        top_hist - np.float64 
        sd - standard deviation over subjects if hist has len >2
    '''
    sd = None
    top = top-1 # python indexing
    if len(hist) >1:
        top_hist = []
        for s in range(len(hist)):
            if len(np.cumsum(hist[s])) >= top+1: 
                top_hist.append(np.cumsum(hist[s])[top])
            elif len(np.cumsum(hist[s])) < top+1:
                top_hist.append(np.cumsum(hist[s])[-1])
        sd = np.std(np.array(top_hist))
        top_hist = np.mean(top_hist) 
    elif len(hist) ==1:
        cumsum = np.cumsum(hist[0], axis=0)
        if len(cumsum) >= top+1:
            top_hist = cumsum[top]
        elif len(cumsum) < top+1:
            top_hist = cumsum[-1]
    if return_sd:
        return top_hist, sd
    else:
        return top_hist

def res2top(filpaths, top):
    # load files
    fls=[]
    for fl in filepaths:
        fls.append(joblib.load(Path(fl)))
    
    # gets histograms for each file for each time window
    hists = []
    for fl in fls:
        hists.append(res2hist(np.array(fl)))

    # get top histograms for each file for each subject
    tops = []
    sds=[]
    for hist in hists:
        top_it, sd_it = hist2top(hist, top, return_sd=True) 
        tops.append(top_it)
        sds.append(sd_it)
    return tops, sds

def topplot(tops, errors, labels=None, xpos=None, title=None):
    
    if xpos==None: 
        xpos=np.arange(0, len(tops), 1, dtype=int)
    
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_axes([0.05,0.05,0.9, 0.88])
    ax.bar(xpos, tops, yerr=errors, color='b', align='center', capsize=10)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_ylim([0,100])
    ax.set_ylabel('Percent ratio')
    fig.suptitle(title)
    return fig, ax

if __name__ =='__main__':
    import argparse
    parser= argparse.ArgumentParser()
    parser.add_argument('-save_fig', action='store_true', default=False, help='Flag, save figs')
    parser.add_argument('-top', type=int, default=1)
    args=parser.parse_args()

    # change if needed
    inp_base=Path('/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/cv_gener/')
    out_path=Path('/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/plots/cv_gener/50hz/')
    title='Top '+str(args.top)+' generic decoding accuracy'
    n_comps=['50', '200', '400']
    prepr='pca'

    filepaths=[]
    labels=[]
    for meth in ["multiviewica","groupica", "permica"]:
        for n_comp in n_comps:
            filepaths.append(inp_base.joinpath(meth, prepr, n_comp, '50hz','time_window13-40', \
                'generic_decoding_results_.pkl'))
            labels.append((meth+'_'+prepr+str(n_comp)))
    tops, sds = res2top(filepaths, args.top)
    fig, ax = topplot(tops, sds, labels, title=title)
    if not args.save_fig:
        plt.show()
    if args.save_fig:
        if not out_path.is_dir():
            out_path.mkdir(parents=True)
        fig.savefig(out_path.joinpath(('top_'+str(args.top)+'_'+prepr+'_'+'_'.join(n_comps)+'.png')), dpi=300)
