#! /bin/bash

#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=regr_fmri_ROI
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=01:00:00
#SBATCH --qos=prio

# N JOBS=18

x_dir='/scratch/akitaitsev/fMRI_Algonautus/activations/cornet_s/PCA1000/'
y_dir_base='/scratch/akitaitsev/fMRI_Algonautus/intersubject_generalization/mini_track/'
out_dir_base='/scratch/akitaitsev/fMRI_Algonautus/regression/mini_track/'
method='PCA'
comp_list=(10 50) 
region_list=('EBA' 'FFA' 'LOC' 'PPA' 'STS' 'V1' 'V2' 'V3' 'V4')

declare -a y_dirs
declare -a out_dirs
declare -a n_comps
declare -a regions

ind=0
for n_comp in ${comp_list[@]}
do
    for region in ${region_list[@]}
    do
        y_dirs[$ind]=$y_dir_base"PCA"$n_comp"/"
        out_dirs[$ind]=$out_dir_base$method$n_comp"/"$region"/"
        n_comps[$ind]=$n_comp
        regions[$ind]=$region
        ((ind=ind+1))
    done
done

y_dirs=${y_dirs[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
n_comps=${n_comps[$SLURM_ARRAY_TASK_ID]}
regions=${regions[$SLURM_ARRAY_TASK_ID]}

echo y_dir: $y_dirs
echo out_dir: $out_dirs
echo n_comp: $n_comps
echo region: $regions

cd /home/akitaitsev/code/intersubject_generalization/fmri/regression/
python linear_regression.py -x_dir $x_dir -y_dir $y_dirs -out $out_dirs -region $regions
