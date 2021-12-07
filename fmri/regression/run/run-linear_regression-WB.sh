#! /bin/bash

#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=regr_fmri-wb
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=01:00:00
#SBATCH --qos=prio

# N JOBS=3

x_dir='/scratch/akitaitsev/fMRI_Algonautus/activations/cornet_s/PCA1000/'
y_dir_base='/scratch/akitaitsev/fMRI_Algonautus/intersubject_generalization/full_track/'
out_dir_base='/scratch/akitaitsev/fMRI_Algonautus/regression/full_track/'
method='PCA'
comp_list=(10 50 100)
region='WB'

declare -a y_dirs
declare -a out_dirs
declare -a n_comps

ind=0
for n_comp in ${comp_list[@]}
do
    y_dirs[$ind]=$y_dir_base"PCA"$n_comp"/"
    out_dirs[$ind]=$out_dir_base$method$n_comp"/"
    n_comps[$ind]=$n_comp
    ((ind=ind+1))
done

y_dirs=${y_dirs[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
n_comps=${n_comps[$SLURM_ARRAY_TASK_ID]}

echo y_dir: $y_dirs
echo out_dir: $out_dirs
echo n_comp: $n_comps

cd /home/akitaitsev/code/intersubject_generalization/fmri/regression/
python linear_regression.py -x_dir $x_dir -y_dir $y_dirs -out $out_dirs -region $region
