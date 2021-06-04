#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=dataset_mat
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=00:10:00
#SBATCH --qos=prio

inp_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/raw_eeg/"
out_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset_matrices/100hz/time_window16-80/"

cd /home/akitaitsev/code/intersubject_generalization/linear/create_featurematrices/ 

echo Creating dataset matrix for time window 16-80
python create_dataset_matrix.py -inp $inp_dir -out $out_dir -srate 100 -time 16 80 
