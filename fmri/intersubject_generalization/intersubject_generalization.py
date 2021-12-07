#! /bin/bash/python3

import numpy as np
import joblib
import argparse
import warnings
from pathlib import Path
from linear_intersubject_generalization_utils import intersubject_generalizer,\
    eigPCA
from sklearn.decomposition import PCA

def load_fmri(base_dir, track, region, dataset, subjs=None):
    '''Loads fMRI data as numpy arrays from Algonautus dataset.
    Returns a tuple of numpy arrays of fmri data and tuple of masks.
    Input:
        base_dir - str or Path object. Top Directory of the dataset.
        track - str, 'full_track' or 'mini_track'
        region - str, region for mini_track(EBA, FFA, LOC, PPA, STS, V1,
            V2, V3, V4, and WB for the whole brain data.
        dataset - str, train or test.
        subjs - list of str with numbers of subject (01, 02, etc.). Defualt=
            None, then using all 10 subjects.
    Outputs:
        fmri - tuple of numpy arrays of fMRI responses for the training set
    '''
    if subjs==None:
        subjs=['01','02','03', '04', '05', '06', '07', '08', '09', '10']
    fmri=[]
    if dataset=='train':
        for subj in subjs:
            fl=joblib.load(Path(base_dir).joinpath("participants_data_v2021", track, \
                ('sub'+subj), (region+'.pkl')))
            fmri.append(fl["train"])
    elif dataset=='test':
        for subj in subjs:
            fl=joblib.load(Path(base_dir).joinpath("participants_data_v2021_test", track, \
                ('sub'+subj), ('organizers_data_'+region+'.pkl')))
            fmri.append(fl["test_data"])
    # average data over repretitions
    if np.any([np.ndim(el)!=3 for el in fmri]):
        raise ValueError('Invalid fMRI data shape.')
    fmri = [np.mean(el, axis=1) for el in fmri]
    return fmri

def auto_search_ncomp(tr_data, te_data, method, npoints=3, start=10, max_n_comp=500):
    '''
    Generate the exp spaced vector of n components for the smart search (takes into account the shape of 
    each data array and determines max acceptible number of components (NB!-method dependent!).
    Inputs:
        tr_data, te_data - 2d numpy arrays of train and test data
        method - sklearn compatilbe object with .fit and .transform methods
        npoints - int, number of points to pick for the search. Defautl=3.
        start - int, staring point of exp spaced numbers for search. Default=10.
    Outputs:
        comps - 1d np array of exp spaced values for the number of components.
    '''
    if method == eig_PCA: 
        end = min([min([el.shape[1] for el in data]) for data in [tr_data, te_data] ])
    else:  
        end = np.amin(np.array([[el.shape for el in data] for data in [tr_data, te_data] ]))
    if end >= max_n_comp:
        warnings.warn('N components detected from exp spaced vecotr is too large ('+str(end)+\
        ') and may result in numerical instability. Resetting end parameter to '+str(max_n_comp)+'.')
        end=max_n_comp
    comps = np.geomspace(start, end, npoints, dtype=int) 
    return comps    



def dim_reduction(data, method, ncomp):
    '''Perform dimentionality reduction.
    Inputs:
        data - llist or tuple of fMRI data arrays for multiple subjects.Each array is a
            3d numpy array of shape (n_videos, n_repetitios, n_voxels),
            on which dim reduction shall be performed.
        method - sklearn compatible object implementing .fit and .transform methods
        ncomp - int, number of components to use (If auto_search=True, ncomp is ignored.
    Output:
        reduced_data - data after dim reduction.
    '''
    if not (callable(getattr(method, 'fit')) or callable(getattr(method, 'transform'))):
        raise ValueError('Mehtod shall implement .fit and .transform methods.')
    method = method(ncomp)
    reduced_data=[]
    for dat in data:
        method.fit(dat)
        reduced_data.append(method.transform(dat))
    # from (n_subj, n_vid, n_feat) to  (n_subj, n_features, n_videos)
    reduced_data = np.transpose(np.stack(reduced_data, axis=0), (0, 2, 1))
    return reduced_data


class eig_PCA(object):
    """
    Wrapper sklearn compatible class for PCA via finding 
    the eigenvectors of the covariance matrix.
    """
    def __init__(self, n_components):
        self.n_components = n_components
        self.tr_data=None
        self.eigval=None
        self.eigvec=None
    def fit(self, data):
        self.tr_data, self.eigval, self.eigvec = eigPCA(data, self.n_components)
    def transform(self, data):
        return self.tr_data    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run intersubject generalization from multiviewica '
    'package. Projecton matrices are learned on train data and then used to project and backproject '
    'the test data.')
    parser.add_argument('-inp', '--input_dir', type=str, help='Top dir of downloaded fMRI data.')
    parser.add_argument('-out','--output_dir', type=str, help='Directory to store projected data, '
    'backprojected data and trained generalized object')
    parser.add_argument('-region', type=str, help='Brain region to work with. WB for whole brain and '
    'EBA, FFA, LOC, PPA, STS, V1, V2, V3, V4 for ROI.')
    parser.add_argument('-method','--method', type=str, help='Which method to use for intersubject '
    'generalization (multiviewica, permica, groupica).', default='multiviewica')
    parser.add_argument('-dim_reduction', type=str, default='PCA', help='Method to reduce '
    'dimensionality of feature matrix before applying intersubjct generalization method. Shall '
    'implement .fit and .transform methods. Assumed to come from sklearn.decomposition module. '
    'Can be=eig_PCA - allows n_comp > min(n_samples, n_features). Defualt=PCA.')
    parser.add_argument('-n_comp','--n_components', type=str, help='Number of components for '
    'dimensionality reduction. Default = 200', default=200)
    parser.add_argument('-auto_search','--auto_comp_search', action='store_true', default=False, help='Do '
    '"smart" search over different number of components automatically determined based on the shape '
    'of the data. ')
    parser.add_argument('-n_points','--n_points_search', type=int, default=3, help='Number of points to test as ncomp '
    'taken from the exponentially spaced set of numbers from 10 to n_features.')
    parser.add_argument('-point', '--point_idx', type=int, default=None, help='Index of point taken from the exponentially'
    'spaced set of numbers from 10 to n_features to be used.')
    parser.add_argument('-max_n_comp', type=int, default=1000, help='Maximal number of components to leave in the data. '
    'Parameter makes sense only with auto_search == True. If automatically defined n_comp is > than max_n_comp, the end '
    ' parameter in np.geomspace() is resetted to max_n_comp.')
    args = parser.parse_args()
     
    # load tarin and test data
    if args.region == 'WB':
        track='full_track'
    else:
        track='mini_track'
    dat_tr = load_fmri(args.input_dir, track, args.region, 'train') 
    dat_te = load_fmri(args.input_dir, track, args.region, 'test') 

    # evaluate dim reduction method
    dim_red_meth = eval(args.dim_reduction) 
    
    # if auto search - determine ncomp
    if args.auto_comp_search:
        start=10
        comps = auto_search_ncomp(dat_tr, dat_te, dim_red_meth, args.n_points_search, start, args.max_n_comp)
        ncomp = comps[args.point_idx]
        print('Component range: '+str(comps))
        print('Chosen ncomp: '+str(ncomp))
    else:
        ncomp = args.n_comp

    # perform dimensionality reduction, get reduced featurematrix of shape (subj, n_features, n_videos)
    dat_tr_red = dim_reduction(dat_tr, dim_red_meth, ncomp)
    dat_te_red = dim_reduction(dat_te, dim_red_meth, ncomp)
    print('Dimensionality reduction completed...')

    # init intersubject generalizer class with user difined method
    mvica_kwargs = {'tol':1e-5, 'max_iter': 10000}

    # do not perform dim reduction in mvica, since it requires 3d array of featurematrices
    # and fmri data has a different number of voxels per subject. PCA shall be performed separately.
    generalizer = intersubject_generalizer(args.method, None, \
        args.dim_reduction.lower(), mvica_kwargs)

    # fit intersubject generalizer on train data, i.e. learn P and W matrices
    generalizer.fit(dat_tr_red)

    shared_train = generalizer.project(dat_tr_red)
    shared_test = generalizer.project(dat_te_red) 

    backprojected_train = generalizer.backproject(shared_train)
    backprojected_test = generalizer.backproject(shared_test)

    # save data
    if args.auto_comp_search:
        output_dir = Path(args.output_dir).joinpath(('PCA'+str(ncomp))) 
    else:
        output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)
    joblib.dump(shared_train, output_dir.joinpath( (str(args.region)+'_shared_train.pkl') ))
    joblib.dump(shared_test, output_dir.joinpath( (str(args.region)+'_shared_test.pkl') ))
