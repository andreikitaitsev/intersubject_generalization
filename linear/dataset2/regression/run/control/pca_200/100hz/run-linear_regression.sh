#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=regr
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=00:10:00
#SBATCH --qos=prio

# N JOBS = 2

# 100 hz veriosn
srate=100
# create inputs
eeg_dir="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/intersubject_generalization/control/pca_200/"$srate"hz/time_window"
dnn_dir="/scratch/akitaitsev/encoding_Ale/dnn_activations/dataset2/"
out_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset2/regression/control/pca_200/"$srate"hz/time_window"

declare -a eeg_dirs
declare -a out_dirs
ind=0
for t in "0-80" "26-80"
do
    eeg_dirs[$ind]=$eeg_dir$t"/av_reps/"
    out_dirs[$ind]=$out_dir$t"/av_reps/"
    ind=$((ind+1))
done

### Extracting the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
eeg_dirs=${eeg_dirs[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}

sleep 10

cd /home/akitaitsev/code/intersubject_generalization/linear/regression/

echo Running linear regression on shared space averaged between subjects
python linear_regression_average.py -eeg_dir $eeg_dirs -dnn_dir $dnn_dir -out_dir $out_dirs

echo Running linear regression on shared space subjectwise 
python linear_regression_subjectwise.py -eeg_dir $eeg_dirs -dnn_dir $dnn_dir -out_dir $out_dirs
