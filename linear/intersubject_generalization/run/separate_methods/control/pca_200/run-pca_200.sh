#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=pca0-40
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=00:05:00
#SBATCH --qos=prio

inp_dir0_40="/scratch/akitaitsev/intersubject_generalizeation/linear/featurematrices/time_window0-40/"
inp_dir13_40="/scratch/akitaitsev/intersubject_generalizeation/linear/featurematrices/time_window13-40/"
out_dir0_40="/scratch/akitaitsev/intersubject_generalizeation/linear/intersubject_generalization/control/pca/200/time_window0-40/"
out_dir13_40="/scratch/akitaitsev/intersubject_generalizeation/linear/intersubject_generalization/control/pca/200/time_window13-40/"
method='PCA'
n_comp=200

cd /home/akitaitsev/code/intersubject_generalization/linear/intersubject_generalization/
echo Reducing dimensions of raw data in time window 0-40
python dim_reduction.py -inp $inp_dir0_40 -out $out_dir0_40 -method $method -n_comp $n_comp

echo Reducing dimensions of raw data in time window 13-40
python dim_reduction.py -inp $inp_dir13_40 -out $out_dir13_40 -method $method -n_comp $n_comp
