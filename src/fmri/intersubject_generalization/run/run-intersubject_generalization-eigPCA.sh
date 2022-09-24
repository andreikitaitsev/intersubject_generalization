#! /bin/bash

#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=MVICA_fmri
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=30:00:00
#SBATCH --qos=prio

# N JOBS = 30
# the script will determine the values of components itself.

inp_dir='/scratch/akitaitsev/fMRI_Algonautus/raw_data/'
out_dir_base='/scratch/akitaitsev/fMRI_Algonautus/intersubject_generalization/'
method='multiviewica'
dim_red='eig_PCA'
region_list=('WB' 'EBA' 'FFA' 'LOC' 'PPA' 'STS' 'V1' 'V2' 'V3' 'V4')
npoints=3
max_n_comp=400

declare -a out_dirs
declare -a points
declare -a regions

ind=0
for point in $(seq 0 $((npoints-1))) #python idx
do
    for region in ${region_list[@]}
    do
        out_dirs[$ind]=$out_dir_base$method"/"$dim_red"/"$region"/"
        points[$ind]=$point
        regions[$ind]=$region
        ((ind=ind+1))
    done
done

out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
regions=${regions[$SLURM_ARRAY_TASK_ID]}
points=${points[$SLURM_ARRAY_TASK_ID]}

echo out_dir: $out_dirs
echo point_idx: $points
echo region: $regions

cd /home/akitaitsev/code/intersubject_generalization/fmri/intersubject_generalization/
python intersubject_generalization.py -inp $inp_dir -out $out_dirs -method $method -dim_reduction $dim_red -region $regions -auto_search -n_points $npoints -point $points -max_n_comp $max_n_comp
