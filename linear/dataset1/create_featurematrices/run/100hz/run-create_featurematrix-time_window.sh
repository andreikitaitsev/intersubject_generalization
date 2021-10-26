#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=featmat
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=00:10:00
#SBATCH --qos=prio

inp_dir0_80="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/dataset_matrices/100hz/time_window0-80/"
inp_dir16_80="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/dataset_matrices/100hz/time_window16-80/"
inp_dir26_80="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/dataset_matrices/100hz/time_window26-80/"
out_dir0_80="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/featurematrices/100hz/time_window0-80/"
out_dir16_80="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/featurematrices/100hz/time_window16-80/"
out_dir26_80="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/featurematrices/100hz/time_window26-80/"

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset1/create_featurematrices/ 

echo Creating feature matrix for time window 0-80
python create_featurematrix.py -inp $inp_dir0_80 -out $out_dir0_80 
echo "Done"

echo Creating feature matrix for time window 0-80
python create_featurematrix.py -inp $inp_dir16_80 -out $out_dir16_80 
echo "Done"

echo Creating feature matrix for time window 0-80
python create_featurematrix.py -inp $inp_dir26_80 -out $out_dir26_80
echo "Done"
