#! /bin/bash
'''Library for the analysis of data from convolutional_autoencoder_raw experiment
'''
import numpy as np
import joblib
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

def _mean_max_acc_over_shuffles(step_dir, nshuffles):
    '''
    Find the maxiaml epta test accuracues over shuffles for one step
    Inputs:
        step_dir - str, the directory where the shuffles are stored for a 
        single step (../stepN/)
        nshuffles - int, number of shuffles
    Outputs: tuples of maximal accs from encoder and decoder layers on different 
        regression types of data averaged over shuffles. 
    (enc_av, enc_av_sd) - encoder, average data
    (dec_av, dec_av_sd) - decoder, average data
    (dec_sw, dec_sw_sd) - decoder, subject-wise data
    '''
    step_dir = Path(step_dir)
    enc_av = []
    enc_sw_mean = []
    enc_sw_sd = []
    dec_av = []
    dec_sw_mean = []
    dec_sw_sd = []
    # load data
    for sh in range(nshuffles):
        accs = joblib.load(step_dir.joinpath(('seed'+str(sh)), \
        'test_accuracies.pkl'))
        enc_av.append(accs["encoder"]["average"])
        dec_av.append(accs["decoder"]["average"])
        dec_sw_mean.append(accs["decoder"]["subjectwise"]["mean"])
        dec_sw_sd.append(accs["decoder"]["subjectwise"]["SD"])
        # so that it works for perceiver
        sw_in_enc =True
        try:
            enc_sw_mean.append(accs["encoder"]["subjectwise"]["mean"])
            enc_sw_sd.append(accs["encoder"]["subjectwise"]["sd"])
        except KeyError:
            sw_in_enc=False
    # select epoch with max average(over shuffles) acc 
    # (step between x samples = epta*index epochs) 
    def _func(acc):
        max_val = np.max(np.mean(np.array(acc), axis=0))
        max_ind = np.argmax(np.mean(np.array(acc), axis=0))
        sd = np.std(np.array(acc)[:,max_ind])
        return max_val, max_ind, sd
    enc_av_mean, _, enc_av_sd = _func(enc_av)
    dec_av_mean, _, dec_av_sd = _func(dec_av)
    dec_sw_mean, maxind, __ = _func(dec_sw_mean)
    dec_sw_sd = np.mean(np.array(dec_sw_sd)[:,maxind])
    if sw_in_enc:
        enc_sw_mean, maxind, _ = _func(enc_sw_mean)
        enc_sw_sd = np.mean(np.array(enc_sw_sd)[:,maxind])
    elif not sw_in_enc:
        enc_sw_mean = None
        enc_sw_sd = None
    return (enc_av_mean, enc_av_sd), (enc_sw_mean, enc_sw_sd), (dec_av_mean, dec_av_sd), \
        (dec_sw_mean, dec_sw_sd)

def create_saturation_profile_data(method_dir, nshuffles, steps):
    '''
    Create saturation profile data for encoder output.
    Inputs:
        method_dir - str, directory where the step directories are stored)
        nshuffles - int, number of shuffles
        steps - list of ints or strs, or generator object, steps for which to calculate saturation 
            profile
    Outputs: 4 dicts with fields 'mean' and 'sd'. Each field contains values for every step of
        saturation profile.
    enc_av
    enc_sw - for conv_autoencoder, is {'mean':None,'sd':None}
    dec_av
    dec_sw
    '''
    enc_av = {'mean': [], 'sd':[]}
    enc_sw = {'mean': [], 'sd':[]}
    dec_av = {'mean':[], 'sd':[]}
    dec_sw = {'mean': [], 'sd': []}
    for step in steps:
        enc_av_tmp, enc_sw_tmp, dec_av_tmp, dec_sw_tmp = _mean_max_acc_over_shuffles(\
            method_dir.joinpath(('step'+str(step))), nshuffles)
        enc_av['mean'].append(enc_av_tmp[0])
        enc_av['sd'].append(enc_av_tmp[1])
        enc_sw['mean'].append(enc_sw_tmp[0])
        enc_sw['sd'].append(enc_sw_tmp[1])
        dec_av['mean'].append(dec_av_tmp[0])
        dec_av['sd'].append(dec_av_tmp[1])
        dec_sw['mean'].append(dec_sw_tmp[0])
        dec_sw['sd'].append(dec_sw_tmp[1])
    return enc_av, enc_sw, dec_av, dec_sw


def saturation_topplot(tops, sds, fig=None, ax=None, labels=None,\
    color='b', graph_type='bar', linestyle='solid', capsize=10, fontsize=12, top=1):
    
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
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.set_ylim([0,100])
    ax.set_xlabel('Ratio of images used for training, %', fontsize=fontsize)
    ax.set_ylabel('Top '+str(top)+' generic decoding accuracy, %', fontsize=fontsize)
    return fig, ax

def plot_saturation_profile(enc_av, enc_sw, dec_av, dec_sw, labels, layer='encoder', title=None, \
    fontsize=12):
    '''Plot saturation profile
    Inputs:
        enc_av, enc_sw, dec_av, dec_sw - dicts, outputs of get_saturation_profile_data.
        layer - output of which conv_autoencoder layer is treated as intersubject EEG
        ('encoder', 'decoder'). Default='encoder' - no subject-wise data.
    Ouptus:
        fig, ax  
    '''
    if title == None:
        title='Top 1 generic decoding accuracy on '+layer+' output'
    if layer == 'encoder' and enc_sw['mean'][0] == None: # if conv_autoencoder
        fig, axes = plt.subplots(1, figsize=(16,9))
        color='C1'
        fig, axes = saturation_topplot(enc_av['mean'], enc_av['sd'], labels=labels,\
            color=color, fig=fig, ax=axes, graph_type='line', capsize=5,\
            fontsize=fontsize)
    elif layer == 'encoder' and enc_sw['mean'][0] != None:
        fig, axes = plt.subplots(1,2, figsize=(16,9))
        colors = ['C1', 'C2']
        linestyles = ['solid', 'dotted']
        ax_titles = ['average', 'subject-wise']
        means = (enc_av['mean'], enc_sw['mean'])
        sds = (enc_av['sd'], enc_sw['sd'])
        for n, ax in enumerate(axes):
            fig, ax = saturation_topplot(means[n], sds[n], labels = labels, \
                color=colors[n], linestyle=linestyles[n],\
                fig=fig, ax=ax, graph_type='line', capsize=5,\
                fontsize=fontsize)
            ax.set_title(ax_titles[n], fontsize=18)
    elif layer == 'decoder':
        fig, axes = plt.subplots(1,2, figsize=(16,9))
        colors = ['C1', 'C2']
        linestyles = ['solid', 'dotted']
        ax_titles = ['average', 'subject-wise']
        means = (dec_av['mean'], dec_sw['mean'])
        sds = (dec_av['sd'], dec_sw['sd'])
        for n, ax in enumerate(axes):
            fig, ax = saturation_topplot(means[n], sds[n], labels = labels, \
                color=colors[n], linestyle=linestyles[n],\
                fig=fig, ax=ax, graph_type='line', capsize=5,\
                fontsize=fontsize)
            ax.set_title(ax_titles[n], fontsize=18)
    fig.suptitle(title, fontsize=18)
    return fig, axes
