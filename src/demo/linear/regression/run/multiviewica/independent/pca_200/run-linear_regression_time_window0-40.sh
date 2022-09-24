#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=linear_regr0-40
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=00:05:00
#SBATCH --qos=prio

# create inputs
eeg_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/intersubject_generalization/multiviewica/independent/time_window0-40/pca_200/"
dnn_dir="/scratch/akitaitsev/encoding_Ale/dnn_activations/dataset1/"
out_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/regression/on_generalized_data/multiviewica/independent/time_window0-40/pca_200/"

sleep 10

cd /home/akitaitsev/code/intersubject_generalization/linear/regression/on_generalized_data/
echo Running linear regression on shared space averaged between subject for time window 0-40
python linear_regression_average.py -eeg_dir $eeg_dir -dnn_dir $dnn_dir -out_dir $out_dir

echo Running linear regression on shared space subjectwise for time window 0-40
python linear_regression_subjectwise.py -eeg_dir $eeg_dir -dnn_dir $dnn_dir -out_dir $out_dir
