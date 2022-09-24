#! /bin/bash

# Create featurematrices with incremental ratio of training images used in steps of 10%

nsteps=10
inp_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/dataset_matrices/50hz/time_window13-40/"
out_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/featurematrices/incremental/50hz/time_window13-40/"

cd /home/akitaitsev/code/intersubject_generalization/linear/create_featurematrices
python create_incremental_featurematrix.py -nsteps $nsteps -inp $inp_dir -out $out_dir
