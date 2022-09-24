#! /bin/bash

#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=MVICA_fmri_WB-test-PCA50
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=30:00:00
#SBATCH --qos=prio

# N JOBS=1

inp_dir='/scratch/akitaitsev/fMRI_Algonautus/raw_data/'
method='multiviewica'
dim_red='PCA'
ncomp=50
region='WB'
out_dir='/scratch/akitaitsev/fMRI_Algonautus/intersubject_generalization/test/full_track/'$method"/"$dim_red"/"$ncomp"/"

echo out_dir $out_dir
cd /home/akitaitsev/code/intersubject_generalization/fmri/intersubject_generalization/
python intersubject_generalization.py -inp $inp_dir -out $out_dir -method $method -dim_reduction $dim_red -n_comp $ncomp -region $region
