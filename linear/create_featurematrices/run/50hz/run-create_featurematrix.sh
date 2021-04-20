#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=featmat
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=00:10:00
#SBATCH --qos=prio

inp_dir0_40="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset_matrices/50hz/time_window0-40/"
inp_dir13_40="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset_matrices/50hz/time_window13-40/"
out_dir0_40="/scratch/akitaitsev/intersubject_generalizeation/linear/featurematrices/50hz/time_window0-40/"
out_dir13_40="/scratch/akitaitsev/intersubject_generalizeation/linear/featurematrices/50hz/time_window13-40/"

cd /home/akitaitsev/code/intersubject_generalization/linear/create_featurematrices/ 

echo Creating feature matrix for time window 0-40
python create_featurematrix.py -inp $inp_dir0_40 -out $out_dir0_40 

echo Creating feature matrix for time window 13-40
python create_featurematrix.py -inp $inp_dir13_40 -out $out_dir13_40 
