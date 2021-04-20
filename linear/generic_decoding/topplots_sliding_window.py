'''Create plots to represent sliding window generic
decoding accuracy on multiviewica with pca and srm'''
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from create_topplots import res2hist, hist2top, topplot
import pandas as pd
import seaborn as sea

def create_time_axis(sliding_window_len, window_start, window_end, sr, epoch_start=-200, epoch_end=600):
    '''Calculate time axis in ms from window in samples with defined srate.
    Inputs:
        sliding_window_len - int, number of samples per sliding window
        window_start - int, start of window used in creation of datasets in samples (e.g. 26)
        window_end - int, end of window used in creation of datasets in samples (e.g. 80)
        sr - int, sampling rate used, Hz 
        epoch_start - float, epoch start relative to stimulus onset, ms
                      Default = -200
        epoch_end - float, epoch end relative to stimulus onset, ms
                    Default = 600
    Outputs:
        timepoints - tuple of floats, end time of each time window, ms
    '''
    sample_spacing= int(1000* (1/sr)) #ms
    ms_per_slidning_window = sliding_window_len*sample_spacing
    n_windows = (window_end - window_start)//sliding_window_len 
    window_start_ms = epoch_start + window_start*sample_spacing
    start = int(window_start_ms + np.round(sample_spacing*sliding_window_len/2))
    timepoints=np.concatenate((np.array([start]),np.tile(sample_spacing*sliding_window_len, (n_windows-1))))
    timepoints=np.cumsum(timepoints)
    return timepoints

def topplots_sliding_window(av_filename, sw_filename, filepaths, labels, top=1,\
    timepoints=None, title=None):
    '''
    Create top plots, i.e. generic decoding results percent ratio for each sliding window.
    Inputs:
        av_filename - str, filename of the avarage regression results
        sw_filename - str, filename of the subjectwise regression results
        filepaths - list of str of directories generic decoding results of different
                    methods/ preprocessing
        labels - list of str, names of methods and preprocessiors (in the same order
                 as filepaths!). 
        top - int
        timepoints - np.array, times of widnows in ms
        title - str, figure title
    Outputs:
        fig, ax - figure and axis habdles

    '''
    # load files
    av_fls=[]
    sw_fls=[]
    for fl in filepaths:
        av_fls.append(joblib.load(Path(fl).joinpath(av_filename)))
        sw_fls.append(joblib.load(Path(fl).joinpath(sw_filename)))
    
    # gets histograms for each file for each time window
    av_hists_cum = []
    sw_hists_cum = []
    for fl in range(len(filepaths)):
        av_hists=[]
        sw_hists=[] 
        # files shall have the same number of windows
        for wind in range(len(av_fls[0])):
            av_hists.append(res2hist(np.array(av_fls[fl][wind])))
            sw_hists.append(res2hist(np.array(sw_fls[fl][wind])))
        av_hists_cum.append(av_hists)
        sw_hists_cum.append(sw_hists)

    # get top histograms for each file for each time window
    av_tops_cum = []
    sw_tops_cum = []
    sw_sds_cum = []
    for fl in range(len(filepaths)):
        av_tops=[]
        sw_tops=[]
        sw_sds=[]
        for wind in range(len(av_fls[0])):
            av_top = hist2top(av_hists_cum[fl][wind], top)
            sw_top, sw_sd = hist2top(sw_hists_cum[fl][wind],top, return_sd=True)
            av_tops.append(av_top)
            sw_tops.append(sw_top)
            sw_sds.append(sw_sd)
        av_tops_cum.append(av_tops)
        sw_tops_cum.append(sw_tops)
        sw_sds_cum.append(sw_sds)
    
    # get decoding accuracy time courses
    av_timecourses=[]
    sw_timecourses=[]
    sds = []
    for fl in range(len(filepaths)):
        av_timecourses.append(np.array(av_tops_cum[fl]))
        sw_timecourses.append(np.array(sw_tops_cum[fl]))
        sds.append(np.array(sw_sds_cum[fl]).squeeze())
    
    if not isinstance(timepoints, np.ndarray) and timepoints==None:
        timepoints = np.linspace(0,len(av_fls[0]), len(av_fls[0]),endpoint=False,\
            dtype=int)
    regr_types = ('av', 'sw')
    methods = labels
    index = pd.MultiIndex.from_product([regr_types, methods], \
        names=['regr_type','method']) 
    ar=np.concatenate((np.array(av_timecourses).T, np.array(sw_timecourses).T), axis=1)
    df=pd.DataFrame(data=ar, index=timepoints, columns=index).unstack()
    df=df.reset_index()
    df.columns=['regr_type','method','timepoint','value']
    
    # add errorbars for subjectwise data
    #sds=np.concatenate((np.full_like(np.concatenate(sds), np.nan), np.concatenate(sds)))
    #df["sds"]=sds
    
    fig, ax = plt.subplots(figsize=(16,9))
    ax=sea.lineplot(data=df, x='timepoint',y='value', style='regr_type', hue='method')
    colors=['b','r'] 
    for num, meth in enumerate(methods):
        ax.errorbar(timepoints, np.array(df.loc[ (df["method"]==meth) & \
            (df["regr_type"]=="sw")]["value"]), sds[num], linestyle='None', color=colors[num], capsize=5)

    #from itertools import cycle
    #linestyles=['-','--','-.',':','<','>',',']
    #if len(linestyles) < len(filepaths):
    #warnings.warn("Number of linestyles to be plotted is less than number of methods to compare."
    #    "Some methods will be plotted in the same linestype.")
    #linecycler=cycle(linestyles)
    
    # plots
    #fig = plt.figure(figsize=(16,9))
    #ax = fig.add_axes([0.05,0.05,0.9, 0.88])
    #for fl, lstyle in zip(range(len(filepaths)), linecycler):
    #    inds=np.linspace(0, len(av_fls[0]), len(av_fls[0]), endpoint=False, dtype=int)
    #    ax.plot(inds, av_timecourses[fl])#, lstyle, 'r')
    #    ax.errorbar(inds, sw_timecourses[fl], yerr=sds[fl])#, linestyle=lstyle, color='b')
    ax.set_ylim([0,100])
    ax.set_xlabel('Middle of time window, ms')
    ax.set_ylabel('Generic decoding results, %')
    ax.set_xticks(timepoints)
    ax.set_xticklabels(timepoints)
    #ax.legend(['average_pca10', 'average_pca50', 'subjectwise_pca10','subjectwise_pca50'])
    fig.suptitle(title)
    return fig, ax


if __name__=='__main__':
    # Topplots pca 16-80
    sw_fname='generic_decoding_results_subjectwise.pkl'
    av_fname='generic_decoding_results_average.pkl'
    filepaths=['/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/sliding_window/multiviewica/main/pca/10/100hz/time_window16-80/',\
        '/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/sliding_window/multiviewica/main/pca/50/100hz/time_window16-80/']
    labels1 = ['pca10', 'pca50']
    timepoints = create_time_axis(5, 16, 80, 100)
    fig1, ax1 = topplots_sliding_window(av_fname, sw_fname, filepaths, labels1, top=1,timepoints=timepoints, \
        title='Generic decoding top 1 results per time window for multiviewica')
    out_path1=Path('/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/plots/sliding_window/multiviewica/100hz/time_window16-80/')
    if not out_path1.is_dir():
        out_path1.mkdir(parents=True)
    fig1.savefig(out_path1.joinpath('top1-pca10_pca50.png'), dpi=300)
    
    # Topplots srm 16-80
    sw_fname='generic_decoding_results_subjectwise.pkl'
    av_fname='generic_decoding_results_average.pkl'
    filepaths=['/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/sliding_window/multiviewica/main/srm/10/100hz/time_window16-80/',\
        '/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/sliding_window/multiviewica/main/srm/50/100hz/time_window16-80/']
    labels2 = ['srm10', 'srm50']
    timepoints = create_time_axis(5, 16, 80, 100) # wind len, wind begin, wind end, sr
    fig2, ax2 = topplots_sliding_window(av_fname, sw_fname, filepaths, labels2, top=1,timepoints=timepoints, \
        title='Generic decoding top 1 results per time window for multiviewica')
    out_path2=Path('/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/plots/sliding_window/multiviewica/100hz/time_window16-80/')
    if not out_path2.is_dir():
        out_path2.mkdir(parents=True)
    fig2.savefig(out_path2.joinpath('top1-srm10_srm50.png'), dpi=300)
    plt.show()
