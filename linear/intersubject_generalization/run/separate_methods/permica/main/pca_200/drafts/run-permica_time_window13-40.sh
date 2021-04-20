#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=permica13-40
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=24:00:00
#SBATCH --qos=prio

### Run multiviewICA on train data and use leared matrices to project and backproject val and test data'''

inp_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/featurematrices/time_window13-40/"
out_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/intersubject_generalization/permica/main/time_window13-40/pca_200/"
method="permica"
echo "Running permica"
sleep 10

cd /home/akitaitsev/code/intersubject_generalization/linear/intersubject_generalization/
# Run the job
python linear_intersubject_generalization_utils.py -inp $inp_dir -out $out_dir -method $method
