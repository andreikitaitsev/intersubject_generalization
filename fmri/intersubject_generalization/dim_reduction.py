#! /bin/env/python
'''Script to reduce data featurematrix dimensionality with sklearn compatible objects
to serve as the control. Defauls =PCA'''
import numpy as np
import copy
import joblib
import argparse
from sklearn.decomposition import PCA
from pathlib import Path
from intersubject_generalization import eig_PCA, dim_reduction, load_fmri

parser = argparse.ArgumentParser(description='Reduces dimesnionality of the featurematrices.'
'package. Projecton matrices are learned on train data and then used to project and backproject '
'val and test data. By default, uses PCA.')
parser.add_argument('-inp', '--input_dir', type=str, help='Directory of feature matrices. Time '
'window is scanned from input directory automatically.')
parser.add_argument('-out','--output_dir', type=str, help='Directory to store projected data, '
'backprojected data and trained generalized object')
parser.add_argument('-dim_reduction', type=str, help='Which method to use for dimensionality '
'reduction. eig_PCA for eigenvalue decomposition PCA. Default = PCA', default='PCA')
parser.add_argument('-n_comp', type=str, help='Number of components for dimensionality '
'reduction method. Default=200.', default=200)
parser.add_argument('-region', type=str, help='Brain region to work with. WB for whole brain and '
'EBA, FFA, LOC, PPA, STS, V1, V2, V3, V4 for ROI.')
args = parser.parse_args()

# evaluate dim reduction method
dim_red_meth = eval(args.dim_reduction) 

# load tarin and test data
if args.region == 'WB':
    track='full_track'
else:
    track='mini_track'
dat_tr = load_fmri(args.input_dir, track, args.region, 'train') 
dat_te = load_fmri(args.input_dir, track, args.region, 'test') 

# perform dimensionality reduction, get reduced featurematrix of shape (subj, n_features, n_videos)
print('Starting dimensionality reduction with PCA with '+str(args.n_comp)+' components...')
dat_tr_red, dat_te_red = dim_reduction(dat_tr, dat_te,  dim_red_meth, args.n_comp)
print('Dimensionality reduction completed...')

# save data
output_dir = Path(args.output_dir)
if not output_dir.is_dir():
    output_dir.mkdir(parents=True)
joblib.dump(dat_tr_red, output_dir.joinpath( (str(args.region)+'_shared_train.pkl') ))
joblib.dump(dat_te_red, output_dir.joinpath( (str(args.region)+'_shared_test.pkl') ))
