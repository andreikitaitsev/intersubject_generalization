#! /bin/bash

#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=regr_fmri_eig_PCA_autoscan
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=01:00:00
#SBATCH --qos=prio

# N JOBS=10

# Note: with auto_scan mode the fmri dirs shall be up to region as well as
# out_dirs. The PCA ncomp subfolder is created automalically.
# Specify the region subfolder manually in the output dir.

x_dir='/scratch/akitaitsev/fMRI_Algonautus/activations/cornet_s/PCA1000/'
y_dir_base='/scratch/akitaitsev/fMRI_Algonautus/intersubject_generalization/comp_search/'
out_dir_base='/scratch/akitaitsev/fMRI_Algonautus/regression/comp_search/'
method='multiviewica'
dim_red='eig_PCA'
region_list=('WB' 'EBA' 'FFA' 'LOC' 'PPA' 'STS' 'V1' 'V2' 'V3' 'V4')

declare -a y_dirs
declare -a out_dirs
declare -a regions

ind=0
for region in ${region_list[@]}
do
    y_dirs[$ind]=$y_dir_base$method"/"$dim_red"/"
    out_dirs[$ind]=$out_dir_base$method"/"$dim_red"/"
    regions[$ind]=$region
    ((ind=ind+1))
done

y_dirs=${y_dirs[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
regions=${regions[$SLURM_ARRAY_TASK_ID]}

echo y_dir: $y_dirs
echo out_dir: $out_dirs
echo region: $regions

cd /home/akitaitsev/code/intersubject_generalization/fmri/regression/
python linear_regression.py -x_dir $x_dir -y_dir $y_dirs -out $out_dirs -region $regions -auto_scan
