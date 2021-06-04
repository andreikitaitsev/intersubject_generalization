#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=regr
#SBATCH --mail-type=end
#SBATCH --mem=2000
#SBATCH --time=00:05:00
#SBATCH --qos=prio

# 16 jobs
dnn_dir="/scratch/akitaitsev/encoding_Ale/dnn_activations/dataset1/"
eeg_dir_base="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/intersubject_generalization/"
out_dir_base="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/regression/"

declare -a eeg_dirs
declare -a out_dirs
declare -a regr_types

for method in "control" "multiviewica" "permica" "groupica"
do
    for prepr in "pca" "srm" 
    do
        for regr_type in "average" "subjectwise"
        do
            regr_types[$ind]=$regr_type
            eeg_dirs[$ind]=$eeg_dir_base"/"$method"/main/"$prepr"/200/time_window13-40/"
            out_dirs[$ind]=$out_dir_base"/"$method"/main/"$prepr"/200/time_window13-40/"
            ((ind=ind+1))
        done
    done
done

echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
eeg_dirs=${eeg_dirs[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
regr_types=${regr_types[$SLURM_ARRAY_TASK_ID]}

cd /home/akitaitsev/code/intersubject_generalization/linear/regression/

echo Running linear regression on gridsearch data 
echo regr type: $regr_types
echo eeg dir $eeg_dirs
echo out dir $out_dirs
python linear_regression.py -eeg_dir $eeg_dirs -dnn_dir $dnn_dir -out_dir $out_dirs -regr_type $regr_types
