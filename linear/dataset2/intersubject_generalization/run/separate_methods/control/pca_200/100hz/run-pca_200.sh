#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=pca0-40
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=00:30:00
#SBATCH --qos=prio

srate=100

inp_dir0_80="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/featurematrices/"$srate"hz/time_window0-80/av_reps/"
inp_dir26_80="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/featurematrices/"$srate"hz/time_window26-80/av_reps/"
out_dir0_80="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/intersubject_generalization/control/pca_200/"$srate"hz/time_window0-80/av_reps/"
out_dir26_80="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/intersubject_generalization/control/pca_200/"$srate"hz/time_window26-80/av_reps/"
method='PCA'
n_comp=200

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset2/intersubject_generalization/
echo Reducing dimensions of raw data in time window 0-80
python dim_reduction.py -inp $inp_dir0_80 -out $out_dir0_80 -method $method -n_comp $n_comp

echo Reducing dimensions of raw data in time window 26-80
python dim_reduction.py -inp $inp_dir26_80 -out $out_dir26_80 -method $method -n_comp $n_comp
