#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=dataset_mat
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=00:10:00
#SBATCH --qos=standard

srate=50
av_reps=True
inp_dir="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/raw_eeg/"
out_dir13_40="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/dataset_matrices/50hz/time_window13-40/2/"

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset2/create_featurematrices/ 

echo Creating dataset matrix for time window 13-40
python create_dataset_matrix.py -inp $inp_dir -out $out_dir13_40 -time 13 40 -srate $srate -av_reps $av_reps
