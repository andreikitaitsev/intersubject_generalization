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
    'topplot_av_sw', 'create_time_axis', 'topplots_sliding_window']

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
        filepaths - list of str, filepaths
        top - int
    Outputs:
        tops - list of top N results
        sds - list of sds over subjects for subject-wise results
    '''
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
    ms_per_slidning_window = sliding_window_len*sample_spacing
    n_windows = (window_end - window_start)//sliding_window_len 
    window_start_ms = epoch_start + window_start*sample_spacing
    start = int(window_start_ms + np.round(sample_spacing*sliding_window_len/2))
    timepoints=np.concatenate((np.array([start]),np.tile(sample_spacing*sliding_window_len, (n_windows-1))))
    timepoints=np.cumsum(timepoints)
    return timepoints


def get_data_slidiing_window(av_filename, sw_filename, filepaths):
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
            av_top, _  = res2top(av_fls[fl][wind], top)
            sw_top, sw_sd = res2top(sw_fls[fl][wind], top)
            av_tops.append(av_top)
            sw_tops.append(sw_top)
            sw_sds.append(sw_sd)
        av_timecourses_mean.append(np.array(av_tops))
        sw_timecourses_mean.append(np.array(sw_tops))
        sw_timecourses_sd.append(np.array(sds).squeeze())
    return av_timecourses_mean, sw_timecourses_mean, sw_timecourses_sd
    

def topplots_sliding_window(av_timecourses_mean, sw_timecourses_mean, sw_timecourses_sd,\
    labels, top=1, timepoints=None, title=None, fontsize=15):
    ''' 
    Create top plots, i.e. generic decoding results percent ratio for each sliding window.
    Inputs:
        labels - list of str, names of methods and preprocessiors (in the same order
                 as filepaths!). 
        top - int
        timepoints - np.array, times of widnows in ms
        title - str, figure title
    Outputs:
        fig, ax - figure and axis habdles
    '''

    if timepoints == None:
        timepoints = np.linspace(0, len(av_fls[0]), len(av_fls[0]),endpoint=False,\
            dtype=int)
    colors=['C1','C2']
    fig, ax = plt.subplots(figsize=(16,9))

    for ind in range(len(labels)):
        # average data
        ax.plot(timepoints, av_timecourses[ind], color=colors[ind], linestyle='solid')
        # subject-wise data
        ax.errorbar(timepoints, sw_timecourses[ind], yerr=sds[ind], color=colors[ind], \
            linestyle='dashed', capsize=10)
    return fig, ax


def topplots_sliding_window_OLD(av_filename, sw_filename, filepaths, labels, top=1,\
    timepoints=None, title=None, plot='matplotlib', fontsize=15):
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
        timepoints = np.linspace(0, len(av_fls[0]), len(av_fls[0]),endpoint=False,\
            dtype=int)
    regr_types = ('av', 'sw')
    methods = labels
    ## matplotlib
    if plot=='matplotlib':
        colors=['C1','C2']
        fig, ax = plt.subplots(figsize=(16,9))
        # average data
        for ind in range(len(methods)):
            ax.plot(timepoints, av_timecourses[ind], color=colors[ind], linestyle='solid')
        # subject-wise data
        for ind in range(len(methods)):
            ax.errorbar(timepoints, sw_timecourses[ind], yerr=sds[ind], color=colors[ind], \
                linestyle='dashed', capsize=10)

    ## seaborn
    elif plot=='seaborn':
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

    ax.set_ylim([0,100])
    ax.set_xlabel('Middle of the sliding time window, ms', fontsize=fontsize)
    ax.set_ylabel('Top 1 generic decoding accuracy, %', fontsize=fontsize)
    ax.set_xticks(timepoints)
    ax.set_xticklabels(timepoints, fontsize=12)
    fig.suptitle(title)
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
    ax.set_ylim([0,100])
    ax.set_xlabel('Ratio of images used for training, %', fontsize=fontsize)
    ax.set_ylabel('Top '+str(top)+' generic decoding accuracy, %', fontsize=fontsize)
    fig.suptitle(title)
    return fig, ax

def plot_saturation_profile(tops_av, sds_av, tops_sw, sds_sw, xtick_labels, labels=None, title=None, top=1,\
    fontsize=12):
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
            color=colors[n], linestyle=linestyles[n],\
            fig=fig, ax=ax, graph_type='line', capsize=5,\
            title=title, fontsize=fontsize)
        if not labels == None:
            ax.legend(labels[n])
        ax.set_title(ax_titles[n], fontsize=18)
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
