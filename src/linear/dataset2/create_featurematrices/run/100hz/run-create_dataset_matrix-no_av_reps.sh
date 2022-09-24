#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=dataset_mat
#SBATCH --mail-type=end
#SBATCH --mem=20000
#SBATCH --time=00:20:00
#SBATCH --qos=standard

srate=100
inp_dir="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/raw_eeg/"
out_dir0_80="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/dataset_matrices/100hz/time_window0-80/no_av_reps/"
out_dir26_80="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/dataset_matrices/100hz/time_window26-80/no_av_reps/"

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset2/create_featurematrices/ 

echo Creating dataset matrix for time window 0-80 with no av_reps
python create_dataset_matrix.py -inp $inp_dir -out $out_dir0_80 -time 0 80 -srate $srate -no_av_reps 

echo Creating dataset matrix for time window 26-80 with no av_reps
python create_dataset_matrix.py -inp $inp_dir -out $out_dir26_80 -time 26 80 -srate $srate -no_av_reps

