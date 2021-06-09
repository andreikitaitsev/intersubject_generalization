#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=dataset_mat
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=00:10:00
#SBATCH --qos=prio

inp_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/raw_eeg/"
out_dir0_40="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/dataset_matrices/50hz/time_window0-40/"
out_dir13_40="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/dataset_matrices/50hz/time_window13-40/"

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset1/create_featurematrices/ 

echo Creating feature matrix for time window 0-40
python create_dataset_matrix.py -inp $inp_dir -out $out_dir0_40 -time 0 40 

echo Creating feature matrix for time window 13-40
python create_dataset_matrix.py -inp $inp_dir -out $out_dir13_40 -time 13 40 
