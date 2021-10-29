#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=sl_win_regr
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=01:00:00
#SBATCH --qos=prio

# N_JOBS = 8
# Run separate linear regression on each sliding window 
dnn_dir="/scratch/akitaitsev/encoding_Ale/dataset1/dnn_activations/"
eeg_dir_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/intersubject_generalization/sliding_window-different_methods/"
out_dir_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/regression/sliding_window-different_methods/"
time_window="16-80"

method_list=("groupica" "multiviewica" "permica" "control")
declare -a eeg_dirs
declare -a out_dirs
declare -a regr_types
declare -a methods

ind=0
for method in ${method_list[@]}
do
    for regr_type in "average" "subjectwise"
    do
        regr_types[$ind]=$regr_type
        eeg_dirs[$ind]=$eeg_dir_base$method"/100hz/time_window"$time_window"/"
        out_dirs[$ind]=$out_dir_base$method"/100hz/time_window"$time_window"/"
        methods[$ind]=$method
        ((ind=ind+1))
    done
done

eeg_dirs=${eeg_dirs[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
regr_types=${regr_types[$SLURM_ARRAY_TASK_ID]}
methods=${methods[$SLURM_ARRAY_TASK_ID]}

echo SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID
echo $methods
echo eeg_dir: $eeg_dirs
echo out_dir: $out_dirs
echo regr_type: $regr_types

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset1/regression/
python linear_regression.py -dnn_dir $dnn_dir -eeg_dir $eeg_dirs -out_dir $out_dirs -regr_type $regr_types -sliding_window 
