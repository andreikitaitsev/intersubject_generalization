#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=featmat
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=00:10:00
#SBATCH --qos=prio

inp_dir0_160="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/dataset_matrices/200hz/time_window0-160/no_av_reps/"
inp_dir52_160="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/dataset_matrices/200hz/time_window52-160/no_av_reps/"
out_dir0_160="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/featurematrices/200hz/time_window0-160/no_av_reps/"
out_dir52_160="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/featurematrices/200hz/time_window52-160/no_av_reps/"

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset2/create_featurematrices/ 

echo Creating feature matrix for time window 0-160 with no av_reps
python create_featurematrix.py -inp $inp_dir0_160 -out $out_dir0_160 

echo Creating feature matrix for time window 52-160 with no av_reps
python create_featurematrix.py -inp $inp_dir52_160 -out $out_dir52_160 
