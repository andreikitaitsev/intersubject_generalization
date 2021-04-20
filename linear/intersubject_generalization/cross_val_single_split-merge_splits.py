#! /env/bin/python3

'''
Megre the output of cross_val_single_split.py (results for each split)
into tuples of shared_train, shared_val, shared_test, each of them containing
lists of data for each CV split (len==n_splits).
This format is compatible with regression and generic decoding functions.
'''
import numpy as np
import joblib
import copy
from pathlib import Path


def megre_splits(inp_dir, n_splits):
    inp_dir=Path(inp_dir)
    shared = ( [], [], [] )
    backprojected = ( [], [], [] )
    data = (shared, backprojected)
    for dtype_num, dtype in enumerate(["shared", "backprojected"]):
        for set_type_num, set_type in enumerate(['_train_','_val_','_test_']):
            for split in range(n_splits):
                fname_it=dtype+set_type+str(split)+'.pkl'
                fpath_it=inp_dir.joinpath(fname_it)
                file_it = joblib.load(fpath_it)
                data[dtype_num][set_type_num].append(file_it)
    generalizers=[]
    for split in range(n_splits):
        fname_it='generalizer'+str(split)+'.pkl'
        fpath_it=inp_dir.joinpath(fname_it)
        file_it = joblib.load(fpath_it)
        generalizers.append(file_it)
    metadata={'split':[]}
    for split in range(n_splits):
        fname_it='metadata'+'_'+str(split)+'.pkl'
        fpath_it=inp_dir.joinpath(fname_it)
        file_it = joblib.load(fpath_it)
        metadata["split"].append(file_it)
    return generalizers, shared, backprojected, metadata

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Megre the outputs of cross_val_single_split.py '
    'into the format compatible with regression and generic decoding functions. Essentially, just merge '
    'individiaul split data in list of individual split data.')
    parser.add_argument('-inp', '--input_dir', type=str, help='Output directory of cross_val_single_split.py.')
    parser.add_argument('-out','--output_dir', type=str, help='Directory to store megred data.')
    parser.add_argument('-n_splits', type=int, default=7, help='N splits in cross validation over subjects.')
    args = parser.parse_args()

    generalizers, shared, backprojected, metadata = megre_splits(args.input_dir, args.n_splits)

    names=['_train.pkl', '_val.pkl', '_test.pkl']
    out_path=Path(args.output_dir)
    for sh_mat, name in zip(shared, names):
        joblib.dump(sh_mat, out_path.joinpath(('shared'+name))) 
    for bpj_mat, name in zip(backprojected, names):
        joblib.dump(bpj_mat, out_path.joinpath(('backprojected'+name)))
    joblib.dump(generalizers, out_path.joinpath('generalizers.pkl'))
    joblib.dump(metadata, out_path.joinpath('metadata.pkl'))
