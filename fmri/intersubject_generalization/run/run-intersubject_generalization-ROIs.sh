#! /bin/bash

#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=MVICA_fmri
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=30:00:00
#SBATCH --qos=prio

# N JOBS=18

inp_dir='/scratch/akitaitsev/fMRI_Algonautus/raw_data/'
out_dir_base='/scratch/akitaitsev/fMRI_Algonautus/intersubject_generalization/mini_track/'
method='PCA'
comp_list=(10 50)
region_list=('EBA' 'FFA' 'LOC' 'PPA' 'STS' 'V1' 'V2' 'V3' 'V4')

declare -a out_dirs
declare -a n_comps
declare -a regions

ind=0
for n_comp in ${comp_list[@]}
do
    for region in ${region_list[@]}
    do
        out_dirs[$ind]=$out_dir_base$method$n_comp"/"
        n_comps[$ind]=$n_comp
        regions[$ind]=$region
        ((ind=ind+1))
    done
done

out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
n_comps=${n_comps[$SLURM_ARRAY_TASK_ID]}
regions=${regions[$SLURM_ARRAY_TASK_ID]}

echo out_dir: $out_dirs
echo n_comp: $n_comps
echo region: $regions

cd /home/akitaitsev/code/intersubject_generalization/fmri/intersubject_generalization/
python intersubject_generalization.py -inp $inp_dir -out $out_dirs -method multiviewica -dim_reduction PCA -n_comp $n_comps -region $regions
