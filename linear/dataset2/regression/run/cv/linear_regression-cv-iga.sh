#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=sl_win_regr
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=01:00:00
#SBATCH --qos=prio

# N_JOBS = 3
# Run separate linear regression on each sliding window 
dnn_dir="/scratch/akitaitsev/encoding_Ale/dataset2/dnn_activations/"
eeg_dir_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/intersubject_generalization/cv/"
out_dir_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/regression/cv/"

method_list=("groupica" "multiviewica" "permica")
declare -a eeg_dirs
declare -a out_dirs

ind=0
for method in ${method_list[@]}
do
    eeg_dirs[$ind]=$eeg_dir_base$method"/100hz/time_window26-80/"
    out_dirs[$ind]=$out_dir_base$method"/100hz/time_window26-80/"
    ((ind=ind+1))
done

eeg_dirs=${eeg_dirs[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
regr_types=${regr_types[$SLURM_ARRAY_TASK_ID]}

echo SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID
echo eeg_dir: $eeg_dirs
echo out_dir: $out_dirs

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset2/regression/
python linear_regression-cv.py -dnn_dir $dnn_dir -eeg_dir $eeg_dirs -out_dir $out_dirs -cv 1
