#! /bin/env/python
'''
Learn projection matrices on arbitrary ratio of the training data.
'''
import joblib
import numpy as np
import argparse
from pathlib import Path
from linear_intersubject_generalization_utils import intersubject_generalizer
from copy import deepcopy

parser = argparse.ArgumentParser(description='Fit intersubject generalizer on incremenal fraction '
'of train data and project the whole train val and test data into shared space.' )
parser.add_argument('-inp', '--input_dir', type=str, help='Directory of feature matrices. ')
parser.add_argument('-out','--output_dir', type=str, help='Directory to store projected data, '
'backprojected data and trained generalized object')
parser.add_argument('-method','--method', type=str, help='Which method to use for intersubject '
'generalization (multiviewica, permica). Groupica is not supported yet.', default='multiviewica')
parser.add_argument('-dim_reduction', type=str, help='Method to reduce dimensionality of feature '
'matrix before applying intersubjct generalization method ("pca" or "srm"). Default = pca', default='pca')
parser.add_argument('-n_comp','--n_components', type=str, help='Number of components for '
'dimensionality reduction. Default = 200', default=200)
parser.add_argument('-ratio', '--training_data_ratio', type=int, default=100, help=
'Ratio of training data to learn prjection matrices on, percents. Default=100.')
parser.add_argument('-seed', type=int, default=1, help=
'Random seed to pick the shuffle the training data with along index dim. Default=1.')
parser.add_argument('-skip_val', action='store_true', default=False, help=
'Flag. If set (true), skip the validation set.')
args = parser.parse_args()

# load tarin test and val feature matrices
featuremat_train = joblib.load(Path(args.input_dir).joinpath('featurematrix_train.pkl'))
if not args.skip_val:
    featuremat_val =  joblib.load(Path(args.input_dir).joinpath('featurematrix_val.pkl'))
featuremat_test = joblib.load(Path(args.input_dir).joinpath('featurematrix_test.pkl'))

# init intersubject generalizer class with user difined method
mvica_kwargs = {'tol':1e-5, 'max_iter': 10000}
if args.n_components=='None':
    args.n_components = None
else:
    args.n_components = int(args.n_components)
gener = intersubject_generalizer(args.method, args.n_components, \
    args.dim_reduction, mvica_kwargs)

# shuffle train data along image dimension
np.random.seed(args.seed)
inds = np.linspace(0, featuremat_train.shape[-1], featuremat_train.shape[-1], endpoint=False, dtype=int)
inds_shuffled = np.random.permutation(inds)
featuremat_train_shuffled = featuremat_train[:,:,inds_shuffled]

# train generalizer on N % of the training data
last_ind= np.round((featuremat_train.shape[-1]/100)*args.trainin_data_ratio)
gener.fit(featuremat_train_shuffled[:, :, :last_ind])

# project and backproject data
shared_tr = gener.project(featuremat_train)
if not args.skip_val:
    shared_val = gener.project(featuremat_val)
shared_test = gener.project(featuremat_test)

back_tr = gener.backproject(shared_tr)
if not args.skip_val:
    back_val = gener.backproject(shared_val)
back_test = gener.backproject(shared_test)

# save data
output_dir = Path(args.output_dir)
if not output_dir.is_dir():
    output_dir.mkdir(parents=True)
joblib.dump(shared_tr, output_dir.joinpath('shared_train.pkl'))
if not args.skip_val:
    joblib.dump(shared_val, output_dir.joinpath('shared_val.pkl'))
joblib.dump(shared_test, output_dir.joinpath('shared_test.pkl'))

joblib.dump(back_tr, output_dir.joinpath('backprojected_train.pkl'))
if not args.skip_val:
    joblib.dump(back_val, output_dir.joinpath('backprojected_val.pkl'))
joblib.dump(back_test, output_dir.joinpath('backprojected_test.pkl'))

joblib.dump(gener, output_dir.joinpath('trained_generalizer.pkl'))

