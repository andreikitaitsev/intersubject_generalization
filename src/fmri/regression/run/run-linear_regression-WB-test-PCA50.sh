#! /bin/bash

#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=regr_fmri-wb
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=01:00:00
#SBATCH --qos=prio

# N JOBS=1

x_dir='/scratch/akitaitsev/fMRI_Algonautus/activations/cornet_s/PCA1000/'
method='multiviewica'
dim_red='PCA'
ncomp=50
region='WB'
y_dir='/scratch/akitaitsev/fMRI_Algonautus/intersubject_generalization/test/full_track/'$method"/"$dim_red"/"$ncomp"/"
out_dir='/scratch/akitaitsev/fMRI_Algonautus/regression/test/full_track/'$method"/"$dim_red"/"$ncomp"/"

cd /home/akitaitsev/code/intersubject_generalization/fmri/regression/
python linear_regression.py -x_dir $x_dir -y_dir $y_dir -out $out_dir -region $region
