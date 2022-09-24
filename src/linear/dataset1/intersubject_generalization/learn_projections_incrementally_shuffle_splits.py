#! /bin/env/python
'''
Learn projection matrices incrementally on train data randomly shuffled for
each step and use them to project the while training data.
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
parser.add_argument('-nsplits', type=int, default=10, help='Number of evenly spaced intervals to divide '
'the train data into. The projection matrices will be learned on this fraction of data.')
parser.add_argument('-step', type=int, help='Split number up to which to concatenate data.')
parser.add_argument('-seed', type=int, help='Seed for random permutation of the train data along image '
'dimension.')
args = parser.parse_args()


# load tarin test and val feature matrices
featuremat_train = joblib.load(Path(args.input_dir).joinpath('featurematrix_train.pkl')) 
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

# (subjs, features, images)
split_train = np.array_split(featuremat_train_shuffled, args.nsplits, axis=-1)  

# fit generalizer on randomly permuted data up to step split
concat = lambda x, ind: np.concatenate(x[0:ind+1], axis = -1)
gener.fit(concat(split_train, args.step))

shared_tr = gener.project(featuremat_train)
shared_val = gener.project(featuremat_val)
shared_test = gener.project(featuremat_test)

back_tr = gener.backproject(shared_tr)
back_val = gener.backproject(shared_val)
back_test = gener.backproject(shared_test)

# save data
output_dir = Path(args.output_dir) 
if not output_dir.is_dir():
    output_dir.mkdir(parents=True)
joblib.dump(shared_tr, output_dir.joinpath('shared_train.pkl'))
joblib.dump(shared_val, output_dir.joinpath('shared_val.pkl'))
joblib.dump(shared_test, output_dir.joinpath('shared_test.pkl'))

joblib.dump(back_tr, output_dir.joinpath('backprojected_train.pkl'))
joblib.dump(back_val, output_dir.joinpath('backprojected_val.pkl'))
joblib.dump(back_test, output_dir.joinpath('backprojected_test.pkl'))

joblib.dump(gener, output_dir.joinpath('trained_generalizer.pkl'))
