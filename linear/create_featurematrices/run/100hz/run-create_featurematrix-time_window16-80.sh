#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=featmat
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=00:10:00
#SBATCH --qos=prio

inp_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset_matrices/100hz/time_window16-80/"
out_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/featurematrices/100hz/time_window16-80/"

cd /home/akitaitsev/code/intersubject_generalization/linear/create_featurematrices/ 

echo Creating feature matrix for time window 16-80
python create_featurematrix.py -inp $inp_dir -out $out_dir 
