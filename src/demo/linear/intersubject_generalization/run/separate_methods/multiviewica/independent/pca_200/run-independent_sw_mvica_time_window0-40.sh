#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=i_ica0-40
#SBATCH --mail-type=end
#SBATCH --mem=2000
#SBATCH --time=02:00:00
#SBATCH --qos=prio

### Run multiviewICA on train data and use leared matrices to project and backproject val and test data'''

inp_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/featurematrices/time_window0-40/"
out_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/intersubject_generalization/multiviewica/independent/time_window0-40/pca_200/"
method="multiviewica"
dim_reduction="pca"
n_components="200"

echo "Running multiviewica"
sleep 10

cd /home/akitaitsev/code/intersubject_generalization/linear/intersubject_generalization/

# Run the job
python independent_subjectwise_multiviewica.py -inp $inp_dir -out $out_dir -method $method -dim_reduction $dim_reduction -n_comp $n_components 
