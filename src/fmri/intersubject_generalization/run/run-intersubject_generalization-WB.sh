#! /bin/bash

#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=MVICA_fmri_WB
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=30:00:00
#SBATCH --qos=prio

# N JOBS=6

inp_dir='/scratch/akitaitsev/fMRI_Algonautus/raw_data/'
out_dir_base='/scratch/akitaitsev/fMRI_Algonautus/intersubject_generalization/full_track/'
method='PCA'
comp_list=(10 50 100 200 400 800)
region='WB'

declare -a out_dirs
declare -a n_comps

ind=0
for n_comp in ${comp_list[@]}
do
    out_dirs[$ind]=$out_dir_base$method$n_comp"/"
    n_comps[$ind]=$n_comp
    ((ind=ind+1))
done

out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
n_comps=${n_comps[$SLURM_ARRAY_TASK_ID]}

echo out_dir: $out_dirs
echo n_comp: $n_comps

cd /home/akitaitsev/code/intersubject_generalization/fmri/intersubject_generalization/
python intersubject_generalization.py -inp $inp_dir -out $out_dirs -method multiviewica -dim_reduction PCA -n_comp $n_comps -region $region
