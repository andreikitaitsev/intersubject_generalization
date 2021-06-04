#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=featmat
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=00:10:00
#SBATCH --qos=prio

inp_dir0_80="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/dataset_matrices/100hz/time_window0-80/av_reps/"
inp_dir26_80="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/dataset_matrices/100hz/time_window26-80/av_reps/"
out_dir0_80="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/featurematrices/100hz/time_window0-80/av_reps/"
out_dir26_80="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/featurematrices/100hz/time_window26-80/av_reps/"

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset2/create_featurematrices/ 

echo Creating feature matrix for time window 0-80 with av_reps
python create_featurematrix.py -inp $inp_dir0_80 -out $out_dir0_80 

echo Creating feature matrix for time window 26-80 with av_reps
python create_featurematrix.py -inp $inp_dir26_80 -out $out_dir26_80 
