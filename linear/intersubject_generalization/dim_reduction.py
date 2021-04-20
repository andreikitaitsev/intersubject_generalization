#! /bin/env/python
'''Script to reduce data featurematrix dimensionality with sklearn compatible objects
to serve as the control. Defauls =PCA'''
import numpy as np
import copy
import joblib
import argparse
import sklearn.decomposition as dec
from pathlib import Path

parser = argparse.ArgumentParser(description='Reduces dimesnionality of the featurematrices.'
'package. Projecton matrices are learned on train data and then used to project and backproject '
'val and test data. By default, uses PCA.')
parser.add_argument('-inp', '--input_dir', type=str, help='Directory of feature matrices. Time '
'window is scanned from input directory automatically.')
parser.add_argument('-out','--output_dir', type=str, help='Directory to store projected data, '
'backprojected data and trained generalized object')
parser.add_argument('-method','--method', type=str, help='Which method to use for dimensionality '
'reduction. Default = PCA', default='PCA')
parser.add_argument('-n_comp','--n_components', type=str, help='Number of components for dimensionality '
'reduction method. Default=200.', default=200)
args = parser.parse_args()

dim_red = eval('dec.'+args.method+'(n_components='+args.n_components+')')

# load tarin test and val feature matrices
featuremat_train = joblib.load(Path(args.input_dir).joinpath('featurematrix_train.pkl'))
featuremat_val =  joblib.load(Path(args.input_dir).joinpath('featurematrix_val.pkl'))
featuremat_test = joblib.load(Path(args.input_dir).joinpath('featurematrix_test.pkl')) 

# pca needs data in form (samples, features)
featuremat_train = np.swapaxes(featuremat_train, 1,2)
featuremat_test = np.swapaxes(featuremat_test, 1,2)
featuremat_val = np.swapaxes(featuremat_val, 1,2)

dim_reds = []
shared_train = []
shared_test = []
shared_val = []
backprojected_train = []
backprojected_test = []
backprojected_val = []
for subj in range(featuremat_train.shape[0]):
    dim_red_iter = copy.deepcopy(dim_red)
    dim_red_iter.fit(featuremat_train[subj,:,:])
    shared_train.append(dim_red_iter.transform(featuremat_train[subj,:,:]))
    shared_test.append(dim_red_iter.transform(featuremat_test[subj,:,:]))
    shared_val.append(dim_red_iter.transform(featuremat_val[subj,:,:]))
    backprojected_train.append(dim_red_iter.inverse_transform(shared_train[-1]))
    backprojected_test.append(dim_red_iter.inverse_transform(shared_test[-1]))
    backprojected_val.append(dim_red_iter.inverse_transform(shared_val[-1]))
    dim_reds.append(dim_red_iter)

output_dir = Path(args.output_dir)
if not output_dir.is_dir():
    output_dir.mkdir(parents=True)

to_proper_shape= lambda x: np.swapaxes(np.squeeze(np.array(x)), 1,2)

joblib.dump(to_proper_shape(shared_train), output_dir.joinpath('shared_train.pkl'))
joblib.dump(to_proper_shape(shared_test), output_dir.joinpath('shared_test.pkl'))
joblib.dump(to_proper_shape(shared_val), output_dir.joinpath('shared_val.pkl'))

joblib.dump(to_proper_shape(backprojected_train), output_dir.joinpath('backprojected_train.pkl'))
joblib.dump(to_proper_shape(backprojected_test), output_dir.joinpath('backprojected_test.pkl'))
joblib.dump(to_proper_shape(backprojected_val), output_dir.joinpath('backprojected_val.pkl'))

joblib.dump(dim_reds, output_dir.joinpath('dim_reds.pkl'))
