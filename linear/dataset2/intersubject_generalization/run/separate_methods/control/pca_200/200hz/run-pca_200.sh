#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=pca0-40
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=02:00:00
#SBATCH --qos=prio

srate=200
inp_dir0_160="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/featurematrices/"$srate"hz/time_window0-160/av_reps/"
inp_dir52_160="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/featurematrices/"$srate"hz/time_window52-160/av_reps/"
out_dir0_160="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/intersubject_generalization/control/pca_200/"$srate"hz/time_window0-160/av_reps/"
out_dir52_160="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/intersubject_generalization/control/pca_200/"$srate"hz/time_window52-160/av_reps/"
method='PCA'
n_comp=200

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset2/intersubject_generalization/
echo Reducing dimensions of raw data in time window 0-160
python dim_reduction.py -inp $inp_dir0_160 -out $out_dir0_160 -method $method -n_comp $n_comp

echo Reducing dimensions of raw data in time window 52-160
python dim_reduction.py -inp $inp_dir52_160 -out $out_dir52_160 -method $method -n_comp $n_comp
