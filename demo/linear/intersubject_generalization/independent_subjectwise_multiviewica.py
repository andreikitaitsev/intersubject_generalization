#! /bin/env/python3
'''Script runs multiviewica independently for every subject
to create a control for comparison with average and subjectwise
multiviewica.
'''

import numpy as np
import joblib 
import linear_intersubject_generalization_utils as ligu
import copy 
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Run intersubject generalization from multiviewica '
'package. Projecton matrices are learned on train data and then used to project and backproject '
'val and test data.')
parser.add_argument('-inp', '--input_dir', type=str, help='Directory of feature matrices. Time '
'window is scanned from directory path automatically.')
parser.add_argument('-out','--output_dir', type=str, help='Directory to store projected data, '
'backprojected data and trained generalized object')
parser.add_argument('-method','--method', type=str, help='Which method to use for intersubject '
'generalization (multiviewica, permica). Groupica is not supported yet.', default='multiviewica')
parser.add_argument('-dim_reduction', type=str, help='Method to reduce dimensionality of feature '
'matrix before applying intersubjct generalization method ("pca" or "srm"). Default = pca', default='pca')
parser.add_argument('-n_comp','--n_components', type=int, help='Number of components for '
'dimensionality reduction. Default = 200', default=200)
args = parser.parse_args()

# load tarin test and val feature matrices
featuremat_train = joblib.load(Path(args.input_dir).joinpath('featurematrix_train.pkl')) 
featuremat_val =  joblib.load(Path(args.input_dir).joinpath('featurematrix_val.pkl')) 
featuremat_test = joblib.load(Path(args.input_dir).joinpath('featurematrix_test.pkl')) # Load data

# create geenralizer call instance
mvica_kwargs = {'tol':1e-5, 'max_iter': 10000, 'verbose': True}
generalizer = ligu.intersubject_generalizer(args.method, args.n_components, \
    args.dim_reduction, mvica_kwargs)

generalizers = []
shared_train = []
shared_test = []
shared_val = []
backprojected_train = []
backprojected_test = []
backprojected_val = []

for subj in range(featuremat_train.shape[0]):
    gener_iter = copy.deepcopy(generalizer)
    gener_iter.fit(np.expand_dims(featuremat_train[subj,:,:], axis=0))
    shared_train.append(gener_iter.project(np.expand_dims(\
        featuremat_train[subj,:,:], axis=0)))
    shared_test.append(gener_iter.project(np.expand_dims(\
        featuremat_test[subj,:,:], axis=0)))
    shared_val.append(gener_iter.project(np.expand_dims(\
        featuremat_val[subj,:,:], axis=0)))
    
    backprojected_train.append(gener_iter.backproject(shared_train[-1]))
    backprojected_test.append(gener_iter.backproject(shared_test[-1]))
    backprojected_val.append(gener_iter.backproject(shared_val[-1]))

    generalizers.append(gener_iter)


# save data
output_dir = Path(args.output_dir)
if not output_dir.is_dir():
    output_dir.mkdir(parents=True)

joblib.dump(np.squeeze(np.array(shared_train)), output_dir.joinpath('shared_train.pkl'))
joblib.dump(np.squeeze(np.array(shared_test)), output_dir.joinpath('shared_test.pkl'))
joblib.dump(np.squeeze(np.array(shared_val)), output_dir.joinpath('shared_val.pkl'))

joblib.dump(np.array(_backprojected_train), output_dir.joinpath('backprojected_train.pkl'))
joblib.dump(np.array(_backprojected_test), output_dir.joinpath('backprojected_test.pkl'))
joblib.dump(np.array(_backprojected_val), output_dir.joinpath('backprojected_val.pkl'))

joblib.dump(generalizers, output_dir.joinpath('trained_generalizers.pkl'))
