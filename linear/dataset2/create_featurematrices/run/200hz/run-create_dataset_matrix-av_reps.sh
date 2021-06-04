#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=dataset_mat
#SBATCH --mail-type=end
#SBATCH --mem=30000
#SBATCH --time=01:00:00
#SBATCH --qos=standard

srate=200
av_reps=True
inp_dir="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/raw_eeg/"
out_dir0_160="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/dataset_matrices/200hz/time_window0-160/av_reps/"
out_dir52_160="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/dataset_matrices/200hz/time_window52-160/av_reps/"

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset2/create_featurematrices/ 

echo Creating dataset matrix for time window 0-160 with av_reps
python create_dataset_matrix.py -inp $inp_dir -out $out_dir0_160 -time 0 160 -srate $srate -av_reps $av_reps

echo Creating dataset matrix for time window 52-160 with av_reps
python create_dataset_matrix.py -inp $inp_dir -out $out_dir52_160 -time 52 160 -srate $srate -av_reps $av_reps
