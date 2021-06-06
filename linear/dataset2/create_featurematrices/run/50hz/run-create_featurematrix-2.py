#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=featmat
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=00:30:00
#SBATCH --qos=prio

inp_dir13_40="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/dataset_matrices/50hz/time_window13-40/2/"
out_dir13_40="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/featurematrices/50hz/time_window13-40/2/"

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset2/create_featurematrices/ 

echo Creating feature matrix for time window 13-40 with av_reps
python create_featurematrix.py -inp $inp_dir13_40 -out $out_dir13_40 
