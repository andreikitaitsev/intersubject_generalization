#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=regr_permica_pca200
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=00:10:00
#SBATCH --qos=prio

dnn_dir="/scratch/akitaitsev/encoding_Ale/dnn_activations/dataset1/"
cd /home/akitaitsev/code/intersubject_generalization/linear/regression/on_generalized_data/

echo Running linear regression in time window 0-40
eeg_dir0_40="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/intersubject_generalization/permica/main/time_window0-40/pca_200/"
out_dir0_40="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/regression/on_generalized_data/permwica/main/time_window0-40/pca_200/"

echo Runnning linear regression on averaged data...
python linear_regression_average.py -eeg_dir $eeg_dir0_40 -dnn_dir $dnn_dir -out_dir $out_dir0_40

echo Running linear regression on subjectwise data...
python linear_regression_subjectwise.py -eeg_dir $eeg_dir0_40 -dnn_dir $dnn_dir -out_dir $out_dir0_40


echo Running linear regression in time window 13-40
eeg_dir13_40="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/intersubject_generalization/permica/main/time_window13-40/pca_200/"
out_dir13_40="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/regression/on_generalized_data/permwica/main/time_window13-40/pca_200/"

echo Runnning linear regression on averaged data...
python linear_regression_average.py -eeg_dir $eeg_dir13_40 -dnn_dir $dnn_dir -out_dir $out_dir13_40

echo Running linear regression on subjectwise data...
python linear_regression_subjectwise.py -eeg_dir $eeg_dir13_40 -dnn_dir $dnn_dir -out_dir $out_dir13_40

