#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=dataset_mat
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=00:10:00
#SBATCH --qos=prio

inp_dir="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/raw_eeg/"
out_dir0_80="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/dataset_matrices/100hz/time_window0-80/"
out_dir16_80="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/dataset_matrices/100hz/time_window16-80/"
out_dir26_80="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/dataset_matrices/100hz/time_window26-80/"

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset1/create_featurematrices/ 

echo Creating feature matrix for time window 0-80
python create_dataset_matrix.py -inp $inp_dir -out $out_dir0_80 -srate 100 -time 0 80 
echo "Done"

echo Creating feature matrix for time window 16-80
python create_dataset_matrix.py -inp $inp_dir -out $out_dir0_80 -srate 100 -time 16 80 
echo "Done"

echo Creating feature matrix for time window 26-80
python create_dataset_matrix.py -inp $inp_dir -out $out_dir26_80 -srate 100 -time 26 80 
echo "Done"
