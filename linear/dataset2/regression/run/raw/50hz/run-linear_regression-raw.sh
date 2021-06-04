#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=regr_raw
#SBATCH --mail-type=end
#SBATCH --mem=2000
#SBATCH --time=00:05:00
#SBATCH --qos=prio

# 4 jobs (0-3)
# 50 hz version

srate=50
dnn_dir="/scratch/akitaitsev/encoding_Ale/dnn_activations/dataset2/"
eeg_dir_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/featurematrices/"$srate"hz/time_window"
out_dir_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/regression/raw/"$srate"hz/time_window"

declare -a eeg_dirs
declare -a out_dirs
declare -a regr_types

ind=0
for time_window in "0-40" "13-40"
do
    for regr_type in "average" "subjectwise"
    do
        eeg_dirs[$ind]=$eeg_dir_base$time_window"/av_reps/"
        out_dirs[$ind]=$out_dir_base$time_window"/av_reps/"
        regr_types[$ind]=$regr_type
        ((ind=ind + 1))
    done
done

eeg_dirs=${eeg_dirs[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
regr_types=${regr_types[$SLURM_ARRAY_TASK_ID]}

echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
echo out_dir: $out_dirs
echo regr type: $regr_types
cd /home/akitaitsev/code/intersubject_generalization/linear/dataset2/regression/

python linear_regression.py -eeg_dir $eeg_dirs -dnn_dir $dnn_dir -out_dir $out_dirs -is_raw "True" -regr_type $regr_types
