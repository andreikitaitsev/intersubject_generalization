#! /bin/env/python
'''
Library to create topplots for generic decoding results for
different paradigms.
'''
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sea
from pathlib import Path

__all__ = ['res2hist', 'hist2top', 'res2top',  'topplot',\
    'topplot_av_sw', 'create_time_axis', 'topplots_sliding_window']

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

def res2top(filepaths, top):
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

def topplot(tops, errors=None, labels=None, xpos=None, title=None, fig=None,\
    ax=None, color='b', graph_type='bar',linestyle='solid', capsize=10, **kwargs):
    
    if isinstance(xpos, type(None)): 
        xpos=np.arange(0, len(tops), 1, dtype=int)
    if fig==None:
        fig = plt.figure(figsize=(16,9))
    if ax==None:
        ax = fig.add_axes([0.05,0.05,0.9, 0.88])
    if graph_type == 'bar':
        ax.bar(xpos, tops, yerr=errors, color=color, align='center', \
            capsize=10, **kwargs)
    elif graph_type=='line':
        ax.errorbar(xpos, tops, yerr=errors, color=color, capsize=capsize, \
            linestyle=linestyle,**kwargs)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_ylim([0,100])
    ax.set_ylabel('Percent ratio')
    fig.suptitle(title)
    return fig, ax

def topplot_av_sw(av_filename, sw_filename, filepaths, top, labels, title=None):
    '''Plots histograms of images being in top n best correlated images 
    (in the row of best correlated original and predicted eeg responses 
    for presented images).
    Inputs:
        av_filename - str, name of generic decoding results on average data
        sw_filename - str, name of generic decodin gresults on subjectwise
                       data
        filepaths - list of strings of each method you would like to compare
        top - int, top position of interest (see res2top)
        labels - list of str, labels for methods for lenegd on the plot (in 
                 the same order as filepaths), first av, then sw
                 (e.g. [av_mvica_pca, av_pca, sw_mvica_pca, sw_pca], etc.)
        title - str, figure title
    Outputs:
        fig - figure handle with top plots
        ax - axis
    '''
    # get histograms
    av_hists = []
    sw_hists = []
    for fl in filepaths:
        av_hists.append(res2hist(np.array(joblib.load(Path(fl).joinpath(av_filename)))))
        sw_hists.append(res2hist(np.array(joblib.load(Path(fl).joinpath(sw_filename)))))
    
    # get top histograms
    av_tops = []
    sw_tops = []
    sw_sds = []
    for fl in range(len(av_hists)): 
        av_top = hist2top(av_hists[fl], top)
        sw_top, sw_sd = hist2top(sw_hists[fl], top, return_sd=True)
        av_tops.append(av_top)
        sw_tops.append(sw_top)
        sw_sds.append(sw_sd)
    
    # plot top histograms 
    errors = np.concatenate((np.full_like(sw_sds, np.nan), np.array(sw_sds)))
    xpos_av=np.arange(0, len(av_tops),1, dtype=int)
    xpos_sw = np.arange(len(av_tops),2*len(av_tops),1, dtype=int)
    #dat4plot = np.concatenate( (np.array(av_tops), np.array(sw_tops)) )
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_axes([0.05,0.05,0.9, 0.88]) 
    ax.bar(xpos_av, av_tops, yerr=errors[xpos_av], color='b', align='center', capsize=10)
    ax.bar(xpos_sw, sw_tops, yerr=errors[xpos_sw], color='r', align='center', capsize=10)
    ax.set_xticks(np.concatenate((xpos_av,xpos_sw)))
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_ylim([0,100])
    ax.set_ylabel('Percent ratio')
    ax.legend(['average','subjectwise'])
    fig.suptitle(title)
    return fig, ax

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
    
    fig, ax = plt.subplots(figsize=(16,9))
    ax=sea.lineplot(data=df, x='timepoint',y='value', style='regr_type', hue='method')
    colors=['b','r'] 
    for num, meth in enumerate(methods):
        ax.errorbar(timepoints, np.array(df.loc[ (df["method"]==meth) & \
            (df["regr_type"]=="sw")]["value"]), sds[num], linestyle='None', color=colors[num], capsize=5)
    ax.set_ylim([0,100])
    ax.set_xlabel('Middle of time window, ms')
    ax.set_ylabel('Generic decoding results, %')
    ax.set_xticks(timepoints)
    ax.set_xticklabels(timepoints)
    #ax.legend(['average_pca10', 'average_pca50', 'subjectwise_pca10','subjectwise_pca50'])
    fig.suptitle(title)
    return fig, ax


if __name__ =='__main__':

    ### Define the data of interest
    time_window = 'time_window13-40'
    methods = ['multiviewica/main/pca/200', 'multiviewica/main/srm/200', 'control/main/pca/200/50hz/']
    labels=['av_mvica_pca','av_mvica_srm','av_pca','sw_mvica_pca', 'sw_mvica_srm','sw_pca']
    
    sw_fname='generic_decoding_results_subjectwise.pkl'
    av_fname='generic_decoding_results_average.pkl'

    project_dir='/scratch/akitaitsev/intersubject_generalizeation/linear/'

    ### create filepaths
    filepaths = []
    for method in methods:
        filepaths.append(Path(project_dir).joinpath('generic_decoding/',\
            method, time_window))
    
    # plot top 1
    fig1, ax1 = topplot_av_sw(av_fname, sw_fname, filepaths, top=1, labels=labels,\
        title='Time window 0 40')
    plt.show()
