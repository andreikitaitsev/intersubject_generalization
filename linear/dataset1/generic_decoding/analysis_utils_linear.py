#! /bin/env/python
'''
Library to for the analysis of experiments with 
linear intersubject algorithms.
'''
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sea
from pathlib import Path

__all__ = ['res2hist', 'hist2top', 'res2top',  'topplot',\
    'topplot_av_sw', 'create_time_axis', 'topplots_sliding_window',\
    'get_data_sliding_window', 'create_time_axis']

### Basic functions (used in all topplots)

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
        if np.min(fl)==np.max(fl):
            hist.append(100)
        if not np.min(fl)==np.max(fl):
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
    '''Converts generic decoding results into top plots.
    Inputs:
        filepaths - list of str, filepaths or list of ints (gen_dec_res)
        top - int
    Outputs:
        tops - list of top N results
        sds - list of sds over subjects for subject-wise results
    '''
    # load files
    if isinstance(filepaths[0], str) or isinstance(filepaths[0], pathlib.PurePath):
        fls=[]
        for fl in filepaths:
            fls.append(joblib.load(Path(fl)))
    else:
        fls=filepaths
    
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

def res2top_single_file(fl, top):
    '''Get top N generic decoding accuracy 
    for a single file of generic decoding results.
    Inputs:
        fl - list of generic decoding results
        top -int, top N
    Outputs:
        top - top N gen dec acc
        sd - SD
    '''
    hist = res2hist(np.array(fl))
    top, sd = hist2top(hist, top, return_sd=True) 
    return top, sd

### Top plot functions

def get_av_sw_data(av_filepaths, sw_filepaths, top):
    '''
    Wrapper function. Accepts average and subjectwise paths and outputs top N values 
    for the average data and means and SDs for subject-wise data.
    Inputs:
        av_filepaths - list of filepaths of gen dec results for average data 
        av_filepaths - list of filepaths of gen dec results for subject-wise data 
        top - int, top position of interest (see res2top)
    Outputs: lists of floats
        av_tops 
        sw_tops
        sw_sds
    '''
    av_tops, _ = res2top(av_filepaths, top)
    sw_tops, sw_sds = res2top(sw_filepaths, top)
    return av_tops, sw_tops, sw_sds

def topplot_av_sw(av_tops, sw_tops, sw_sds, labels, top=1, hatches=None, title=None,\
    plot_values=False, xtick_rotation=None, fontsize=12):
    '''
    Plots histograms of images being in top n best correlated images 
    (in the row of best correlated original and predicted eeg responses 
    for presented images).
    Inputs:
        av_tops - list of top N accuracies for average data. Outputs of res2top funciton
        sw_tops - list of top N accuracies for subject-wise data. Outputs of res2top funciton
        sw_sds - list of SDs of top N accuracies for subject-wise data. Outputs of res2top funciton
        labels - list of str, labels for methods in the order they are packed into av_tops.
        top - int
        hatches - list of strs, hatches of control and intesubject generalization algorithms (IGAs)
            (1st entry - control, 2nd - IGAs)
        title - str, figure title
        plot_values - bool, whether to plot result values on the graph. 
            Default=False.
    Outputs:
        fig - figure handle with top plots
        ax - axis
    '''
    # plot top barplots
    av_tops = np.array(av_tops)
    sw_tops=np.array(sw_tops)
    sw_sds=np.array(sw_sds)
    x=np.linspace(0, len(av_tops), len(av_tops), endpoint=False, dtype=int)
    fig, axes = plt.subplots(1,2, figsize=(16,9))
    if not hatches==None:
        ctrl_idx=np.array((0,1),dtype=int)
        iga_idx=np.setdiff1d(x, ctrl_idx).astype(int)
        axes[0].bar(x[ctrl_idx], av_tops[ctrl_idx], color='C1', align='center', capsize=10, hatch=hatches[0])
        axes[0].bar(x[iga_idx], av_tops[iga_idx], color='C1', align='center', capsize=10, hatch=hatches[1])

        axes[1].bar(x[ctrl_idx], sw_tops[ctrl_idx], yerr=sw_sds[ctrl_idx], color='C2', align='center', capsize=10, hatch=hatches[0])
        axes[1].bar(x[iga_idx], sw_tops[iga_idx], yerr=sw_sds[iga_idx], color='C2', align='center', capsize=10, hatch=hatches[1])
    else:
        axes[0].bar(x, av_tops, color='C1', align='center', capsize=10)
        axes[1].bar(x, sw_tops, yerr=sw_sds, color='C2', align='center', capsize=10)
    if plot_values:
        for i,v in enumerate(av_tops):
            axes[0].text(x[i]+0.025, v+1, '{:.0f}'.format(v), color='C1')
        for i, v in enumerate(sw_tops):
            axes[1].text(x[i]+0.025, v+1, '{:.0f}'.format(v), color='C2')
            axes[1].text(x[i]+0.025, v+4, '{:.0f}'.format(v), color='C2')
    for ax in axes:
        ax.set_xticks(x) 
        ax.set_xticklabels(labels, rotation=xtick_rotation, fontsize=fontsize)
        ax.set_ylim(0,100)
        ax.set_ylabel('Top '+str(top)+'generic decoding accuracy, %', fontsize=fontsize)
    axes[0].set_title('average', fontsize=fontsize)
    axes[1].set_title('subject-wise', fontsize=fontsize)
    fig.suptitle(title, fontsize=fontsize)
    return fig, axes


### Time resolution/ Sliding window

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
    samples_per_window=window_end-window_start+1
    n_windows = (window_end - window_start)//sliding_window_len 
    window_start_ms = epoch_start + window_start*sample_spacing
    start = int(window_start_ms + np.ceil(sample_spacing*sliding_window_len/2))
    timepoints=np.concatenate((np.array([start]),np.tile(sample_spacing*sliding_window_len, (n_windows-1))))
    timepoints=np.cumsum(timepoints)
    return timepoints


def get_data_sliding_window(av_filename, sw_filename, filepaths, top=1):
    '''
    Create timecourses of data for sliding time widnow experiment.
    Inputs:
        av_filename - str, filename of the avarage regression results
        sw_filename - str, filename of the subjectwise regression results
        filepaths - list of str of directories generic decoding results of different
                    methods/ preprocessing
    Ouptus: lists of dec dec acc values for each time window position
        av_timecourses_mean
        sw_timecourses_mean
        sw_timecourses_sd 
    '''
    # load files
    av_fls=[]
    sw_fls=[]
    for fl in filepaths:
        av_fls.append(joblib.load(Path(fl).joinpath(av_filename)))
        sw_fls.append(joblib.load(Path(fl).joinpath(sw_filename)))
    
    # get top 1 accuracy for each file (method) in each time window
    av_timecourses_mean = []
    sw_timecourses_mean = []
    sw_timecourses_sd = []
    for fl in range(len(filepaths)):
        av_tops=[]
        sw_tops=[]
        sw_sds=[]
        for wind in range(len(av_fls[0])):
            av_top, _  = res2top_single_file(av_fls[fl][wind], top)
            sw_top, sw_sd = res2top_single_file(sw_fls[fl][wind], top)
            av_tops.append(av_top)
            sw_tops.append(sw_top)
            sw_sds.append(sw_sd)
        av_timecourses_mean.append(np.array(av_tops))
        sw_timecourses_mean.append(np.array(sw_tops))
        sw_timecourses_sd.append(np.array(sw_sds).squeeze())
    return av_timecourses_mean, sw_timecourses_mean, sw_timecourses_sd
    

def topplots_sliding_window(av_timecourses_mean, sw_timecourses_mean, sw_timecourses_sd,\
    labels, top=1, timepoints=None, title=None, fontsize=15, linestyles=None, ylim=(0,100)):
    ''' 
    Create top plots, i.e. generic decoding results percent ratio for each sliding window.
    Note, that the maxismal number of methods is 4 (check the funciton to see, why)
    Inputs:
        av_timecourses_mean - list of mean timecourses for average data
        sw_timecourses_mean - list of mean timecourses for subject-wise data
        sw_timecourses_sd - list of SDs of timecourses for subject-wise data
        labels - list of str, names of methods and preprocessiors (in the same order
                 as filepaths!). 
        top - int
        timepoints - np.array, times of widnows in ms
        title - str, figure title
        linestyles - list of str, linestyles for different methods (e.g. so that 
            control is dashed and IGA are solid) Default=None
        ylim - list or tuple of min and max values on the Y axis. Default=[0,100].
    Outputs:
        fig, ax - figure and axis habdles
    '''

    if timepoints is None:
        timepoints = np.linspace(0, len(av_fls[0]), len(av_fls[0]),endpoint=False,\
            dtype=int)
    colors=['C1','C2', 'C3', 'C4']
    pane_labels=['average', 'subject-wise']
    ylabel='Top 1 generic decoding accuracy, %'
    xlabel='Middle of the sliding time window, ms'
    fig, axes = plt.subplots(1, 2, figsize=(16,9))
    if not title is None: 
        fig.suptitle(title, fontsize=18) 
    if linestyles is None:
        linestyles = ['solid', 'solid', 'solid', 'solid'] 
    # average
    for ind in range(len(labels)):
        axes[0].plot(timepoints, av_timecourses_mean[ind], color=colors[ind], linestyle=linestyles[ind])
    # subject-wise
    for ind in range(len(labels)):
        axes[1].errorbar(timepoints, sw_timecourses_mean[ind], yerr=sw_timecourses_sd[ind], color=colors[ind], \
            linestyle=linestyles[ind], capsize=10)
    for n, ax in enumerate(axes):
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        ax.set_ylim([ylim[0], ylim[1]])
        ax.legend(labels, fontsize=12)
        ax.set_title(pane_labels[n], fontsize=16)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
    return fig, ax


### Saturation profile

def average_shuffles(meth_dir, av_filename, sw_filename, steps, nshuffles, top=1):
    '''Average gen dec accuracies over shuffles for every step.
    Inputs:
        meth_dir - str, directory where the shuffles are stored.
        steps - list of ints or strs, steps (ratios of training images used)
        nshuffles - int, number of shuffles used
    Outputs:
        lists of floats
        tops_av - top n gen dec accs averaged iver shuffles for average data
        sds_av - SD over shuffles of average data
        tops_sw - top n gen dec accs averaged over shuffles for subject-wise data
        sds_av - SD over shuffles of subject-wise data
    '''
    tops_av = []
    sds_av = []
    tops_sw = []
    sds_sw = []
    for num, step in enumerate(steps): 
        fpaths_av = []
        fpaths_sw = []
        for shuffle in range(nshuffles): #np.linspace(0, nshuffles, nshuffles, endpoint=False, dtype=int):
            fpaths_av.append(Path(meth_dir).joinpath( ('shuffle_'+str(shuffle)), \
                ('step_'+str(step)), av_filename))
            fpaths_sw.append(Path(meth_dir).joinpath( ('shuffle_'+str(shuffle)), \
                ('step_'+str(step)), sw_filename))
        # gen dec results for every shuffle in step N
        tops_av_it, _ = res2top(fpaths_av, top)
        tops_sw_it, sds_sw_it = res2top(fpaths_sw, top)
        
        ## mean and sd over shuffles for step N
        tops_av.append(np.mean(np.array(tops_av_it)))
        sds_av.append(np.std(np.array(tops_av_it)))
        tops_sw.append(np.mean(np.array(tops_sw_it)))
        sds_sw.append(np.mean(np.array(sds_sw_it)))
    return tops_av, sds_av, tops_sw, sds_sw

def _saturation_topplot(tops, sds, fig=None, ax=None, xtick_labels=None, xpos=None, title=None, \
    color='b', graph_type='bar',linestyle='solid', capsize=10, fontsize=12, top=1):
    '''Helper function for plot_saturation_profile'''
    if isinstance(xpos, type(None)): 
        xpos=np.arange(0, len(tops), 1, dtype=int)
    if fig==None:
        fig = plt.figure(figsize=(16,9))
    if ax==None:
        ax = fig.add_axes([0.05,0.05,0.9, 0.88])
    if graph_type == 'bar':
        ax.bar(xpos, tops, yerr=sds, color=color, align='center', \
            capsize=10, **kwargs)
    elif graph_type=='line':
        ax.errorbar(xpos, tops, yerr=sds, color=color, capsize=capsize, \
            linestyle=linestyle)
    ax.set_xticks(xpos)
    ax.set_xticklabels(xtick_labels, fontsize=fontsize)
    ax.set_ylim([40,102])
    ax.set_xlabel('Ratio of images used for training, %', fontsize=fontsize)
    ax.set_ylabel('Top '+str(top)+' generic decoding accuracy, %', fontsize=fontsize)
    fig.suptitle(title)
    return fig, ax

def plot_saturation_profile(tops_av, sds_av, tops_sw, sds_sw, xtick_labels, xpos=None, fig=None, ax=None, \
    labels=None, title=None, top=1, fontsize=12):
    '''
    Plot saturation profile.
    Inputs:
        tops_av - list of top N generic decoding accuracies on average data averaged over shuffles
        sds_av - list of standard deviations of top N gen dec accs on average data over shuffles
        tops_sw - list of top N gen dec accs averaged on subjectwise data averaged over shuffles
        sds_sw - list of SDs of top N gen dec accs over subjects avegared over shuffles
        xtick_labels - list of steps, ratios of training images used, %
        labels - list of str, names of different methods to plot
        title - str
        top -int, default=1
        fontsize - int
    Outputs:
        fig, ax - figure and axes handle for saturation profile plot
    '''
    tops = (tops_av, tops_sw)
    sds = (sds_av, sds_sw)
    fig, axes = plt.subplots(1,2, figsize=(16,9))
    colors = ['C1', 'C2']
    linestyles = ['solid', 'dotted']
    ax_titles = ['average', 'subject-wise']
    for n, ax in enumerate(axes):
        fig, ax = _saturation_topplot(tops[n], sds[n], xtick_labels = xtick_labels, \
            xpos=xpos, color=colors[n], linestyle=linestyles[n],\
            fig=fig, ax=ax, graph_type='line', capsize=5,\
            title=title, fontsize=fontsize)
        if not labels == None:
            ax.legend(labels[n])
        ax.set_title(ax_titles[n], fontsize=18)
    return fig, axes

def plot_saturation_profile_different_methods(tops_av, sds_av, tops_sw, sds_sw, xtick_labels,\
    xpos, fig=None, ax=None, labels=None, title=None, top=1, fontsize=12):
    '''
    Plot saturation profile.
    Inputs:
        tops_av - list of top N generic decoding accuracies on average data averaged over shuffles
            FOR DIFFERENT METHODS
        sds_av - list of standard deviations of top N gen dec accs on average data over shuffles
            FOR DIFFERENT METHODS
        tops_sw - list of top N gen dec accs averaged on subjectwise data averaged over shuffles
            FOR DIFFERENT METHODS
        sds_sw - list of SDs of top N gen dec accs over subjects avegared over shuffles
            FOR DIFFERENT METHODS
        xtick_labels - list of steps, ratios of training images used, %
        xpos - list of x positions for each method for the bar plots
        labels - list of str, names of different methods to plot
        title - str
        top -int, default=1
        fontsize - int
    Outputs:
        fig, ax - figure and axes handle for saturation profile plot
    '''
    tops = (tops_av, tops_sw)
    sds = (sds_av, sds_sw)
    fig, axes = plt.subplots(1,2, figsize=(16,9))
    colors = ['C1', 'C2', 'C3']
    linestyles = ['solid', 'dotted']
    ax_titles = ['average', 'subject-wise']
    for meth in range(len(tops_av)):
        for n, ax in enumerate(axes):
            fig, ax = _saturation_topplot(tops[n][meth], sds[n][meth], xtick_labels = xtick_labels, \
                xpos=xpos[meth], color=colors[meth], linestyle=linestyles[n],\
                fig=fig, ax=ax, graph_type='line', capsize=5,\
                title=title, fontsize=fontsize)
            ax.set_title(ax_titles[n], fontsize=18)
    if not labels==None:
        for ax in axes:
            ax.legend(labels)
    return fig, axes

# Demo
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
