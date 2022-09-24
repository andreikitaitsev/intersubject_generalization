#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=regr
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=00:05:00
#SBATCH --qos=prio

# create inputs
eeg_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/intersubject_generalization/control/pca/200/time_window"
dnn_dir="/scratch/akitaitsev/encoding_Ale/dnn_activations/dataset1/"
out_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/regression/control/pca/200/time_window"

declare -a eeg_dirs
declare -a out_dirs
ind=0
for t in "0-40" "13-40"
do
    eeg_dirs[$ind]=$eeg_dir$t
    out_dirs[$ind]=$out_dir$t
    ind=$((ind+1))
done

### Extracting the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
eeg_dirs=${eeg_dirs[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}

sleep 10

cd /home/akitaitsev/code/intersubject_generalization/linear/regression/

echo Running linear regression on shared space averaged between subject for time window 0-40
python linear_regression_average.py -eeg_dir $eeg_dirs -dnn_dir $dnn_dir -out_dir $out_dirs

echo Running linear regression on shared space subjectwise for time window 0-40
python linear_regression_subjectwise.py -eeg_dir $eeg_dirs -dnn_dir $dnn_dir -out_dir $out_dirs
