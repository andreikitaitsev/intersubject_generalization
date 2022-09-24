#! /bin/bash

#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=regr_fmri-wb
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=01:00:00
#SBATCH --qos=prio

# Run regression on x obtained with different n components
# N JOBS=9

track='full_track'
method='multiviewica'
x_dim_red='PCA'
x_n_comp_list=(100 500 1000)
y_dim_red='PCA'
x_dir_base='/scratch/akitaitsev/fMRI_Algonautus/dnn_features/cornet_s/'
y_dir_base='/scratch/akitaitsev/fMRI_Algonautus/intersubject_generalization/'$method'/full_track/'
out_dir_base='/scratch/akitaitsev/fMRI_Algonautus/regression/'$method'/'$track'/'
comp_list=(10 50 100)
region='WB'

declare -a x_dirs
declare -a y_dirs
declare -a out_dirs
declare -a n_comps
declare -a x_n_comps

ind=0
for x_n_comp in ${x_n_comp_list[@]}
do
    for n_comp in ${comp_list[@]}
    do
        x_dirs[$ind]=$x_dir_base$x_dim_red$x_n_comp"/"
        y_dirs[$ind]=$y_dir_base"/"$y_dim_red$n_comp"/"
        out_dirs[$ind]=$out_dir_base"/x_"$x_dim_red$x_n_comp"/"$method"/"$y_dim_red$n_comp"/"
        n_comps[$ind]=$n_comp
        x_n_comps[$ind]=$x_n_comp
        ((ind=ind+1))
    done
done

y_dirs=${y_dirs[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
n_comps=${n_comps[$SLURM_ARRAY_TASK_ID]}
x_n_comps=${x_n_comps[$SLURM_ARRAY_TASK_ID]}

echo y_dir: $y_dirs
echo out_dir: $out_dirs
echo x_n_comp: $x_n_comps
echo n_comp: $n_comps

cd /home/akitaitsev/code/intersubject_generalization/fmri/regression/
python linear_regression.py -x_dir $x_dirs -y_dir $y_dirs -out $out_dirs -region $region
