#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=dataset_mat
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=00:10:00
#SBATCH --qos=standard

srate=50
inp_dir="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/raw_eeg/"
out_dir0_40="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/dataset_matrices/50hz/time_window0-40/no_av_reps/"
out_dir13_40="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/dataset_matrices/50hz/time_window13-40/no_av_reps/"

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset2/create_featurematrices/ 

echo Creating feature matrix for time window 0-40
python create_dataset_matrix.py -inp $inp_dir -out $out_dir0_40 -time 0 40 -srate $srate -no_av_reps 

echo Creating feature matrix for time window 13-40
python create_dataset_matrix.py -inp $inp_dir -out $out_dir13_40 -time 13 40 -srate $srate -no_av_reps
