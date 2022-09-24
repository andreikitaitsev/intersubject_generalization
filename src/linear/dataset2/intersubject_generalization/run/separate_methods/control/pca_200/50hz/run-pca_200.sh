#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=pca0-40
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=00:30:00
#SBATCH --qos=prio

srate=50

inp_dir0_40="/scratch/akitaitsev/intersubject_generalization/linear/dataset2//featurematrices/"$srate"hz/time_window0-40/av_reps/"
inp_dir13_40="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/featurematrices/"$srate"hz/time_window13-40/av_reps/"
out_dir0_40="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/intersubject_generalization/control/pca_200/"$srate"hz/time_window0-40/av_reps/"
out_dir13_40="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/intersubject_generalization/control/pca_200/"$srate"hz/time_window13-40/av_reps/"
method='PCA'
n_comp=200

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset2/intersubject_generalization/
echo Reducing dimensions of raw data in time window 0-40
python dim_reduction.py -inp $inp_dir0_40 -out $out_dir0_40 -method $method -n_comp $n_comp

echo Reducing dimensions of raw data in time window 13-40
python dim_reduction.py -inp $inp_dir13_40 -out $out_dir13_40 -method $method -n_comp $n_comp
