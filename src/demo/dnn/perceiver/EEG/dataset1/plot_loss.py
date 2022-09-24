#! /bin/env/python3

import numpy as np
import matplotlib.pyplot as plt

def unroll_loss(losses):
    epochwise_loss = []
    all_loss = []
    for n_epoch in range(len(losses)):
        epoch = losses["epoch"+str(n_epoch)]
        epochwise_loss.append(np.mean(epoch))
        all_loss = all_loss+ epoch
    return epochwise_loss, all_loss

def plot_losses(loss_list, fig=None, ax=None, legend=None, suptitle=None):
    '''
    Inputs:
        loss_list - list of lists of losses from different network configs
    Ouputs:
        fig, ax
    '''
    if fig == None:
        fig, ax = plt.subplots()
    for loss in loss_list:    
        ax.plot(loss)
    if legend != None:
        ax.legend(legend)
    if suptitle !=None:
        fig.suptitle(suptitle)
    return fig, ax

if __name__ == '__main__':
    import argparse
    import joblib
    import os
    parser=argparse.ArgumentParser(description="Plot losses.")
    parser.add_argument('-dir', type=str, help='Directory to scan for loss files.')
    parser.add_argument('-no_average', action='store_true', default=False, help='If tire, do not average all losses '
    'for 1 epoch. Default = False.')
    parser.add_argument('-out_dir', type=str, default=None, help='Directory to save figures. Default=None')
    args=parser.parse_args()
    
    dir_base = args.dir
    # scan dir for loss files
    loss_list = []
    legend = []
    for root, dirs, files in os.walk(dir_base, topdown=True):
        if 'losses.pkl'in files:
            loss_list.append(joblib.load(os.path.join(root, 'losses.pkl')))
            legend.append(str(os.path.split(root)[-1]))
    
    # unroll losses
    all_loss_list = []
    mean_loss_list= []
    for loss in loss_list:
        mean_loss, all_loss = unroll_loss(loss)
        all_loss_list.append(all_loss)
        mean_loss_list.append(mean_loss)

    # plot loss files  
    if not  args.no_average:
        fig, ax = plot_losses(mean_loss_list, legend=legend)
    elif args.no_average:
        fig, ax = plot_losses(all_loss_list, legend=legend)

    plt.show()
    if not args.out_dir == None:
        if not os.path.isdir(args.out_dir):
            os.path.makedirs(args.out_dir)
        fig.savefig(args.out_dir)
