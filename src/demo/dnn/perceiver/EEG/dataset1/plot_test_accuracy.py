#! /bin/env/python3

import numpy as np
import matplotlib.pyplot as plt


def plot_acc(acc_list, fig=None, ax=None, legend=None, suptitle=None):
    '''
    Inputs:
        acc_list - list of lists of test_accuracies from different network configs
    Ouputs:
        fig, ax
    '''
    if fig == None:
        fig, axes = plt.subplots(2,2)
    

    for acc in acc_list:
        acc_enc = acc["encoder"]
        acc_ph = acc["projection_head"]

        x = np.linspace(1, len(acc_enc["average"]), len(acc_enc["average"]), endpoint=True)
        assert len(acc_enc["subjectwise"]["mean"]) == len(acc_enc["subjectwise"]["SD"]) == len(acc_ph["average"])==\
        len(acc_ph["subjectwise"]["mean"]) == len( acc_ph["subjectwise"]["SD"])
        
        ## encoder 
        # average
        axes[0,0].plot(x, acc_enc["average"])
        #subjectwise
        axes[1,0].errorbar(x, acc_enc["subjectwise"]["mean"], acc_enc["subjectwise"]["SD"])

        ## ph
        # average
        axes[0,1].plot(x, acc_ph["average"])
        # subjectwise
        axes[1,1].errorbar(x, acc_ph["subjectwise"]["mean"],acc_ph["subjectwise"]["SD"])
    if legend != None:
        axes[0,0].legend(legend)
        axes[0,1].legend(legend)
        axes[1,0].legend(legend)
        axes[1,1].legend(legend)
    axes[0,0].set_title('encoder_average')
    axes[1,0].set_title('encoder_subjectwise')
    axes[0,1].set_title('ph_average')
    axes[1,1].set_title('ph_subjectwise')

    if not suptitle==None:
        fig.suptitle(suptitle)
    return fig, ax

if __name__ == '__main__':
    import argparse
    import joblib
    import os
    parser=argparse.ArgumentParser(description="Plot test accuracies.")
    parser.add_argument('-dir', type=str, help='Directory to scan for test_acc files.')
    parser.add_argument('-out_dir', type=str, default=None, help='Directory to save figures. Default=None')
    args=parser.parse_args()
    
    dir_base = args.dir
    # scan dir for acc files
    acc_list = []
    legend = []
    for root, dirs, files in os.walk(dir_base, topdown=True):
        if 'test_accuracies.pkl'in files:
            acc_list.append(joblib.load(os.path.join(root, 'test_accuracies.pkl')))
            legend.append(str(os.path.split(root)[-1]))
    
    
    # plot loss files  
    fig, ax = plot_acc(acc_list, legend=legend)

    plt.show()
    if not args.out_dir == None:
        if not os.path.isdir(args.out_dir):
            os.path.makedirs(args.out_dir)
        fig.savefig(args.out_dir)
