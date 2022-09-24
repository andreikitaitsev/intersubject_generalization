#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=gica_pca200
#SBATCH --mail-type=end
#SBATCH --mem=60000
#SBATCH --time=20:00:00
#SBATCH --qos=prio

### 50 hz 

### Run groupica with pca=200 on train data and use leared matrices to project 
### and backproject val and test data on 
# data with averaged repetitions

srate=50
inp_dir="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/featurematrices/"$srate"hz/time_window13-40/av_reps/"
out_dir="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/intersubject_generalization/groupica/pca_200/"$srate"hz/time_window13-40/av_reps/"

method="groupica"
dim_reduction="pca"
n_components="200"

echo "Running groupica"
cd /home/akitaitsev/code/intersubject_generalization/linear/dataset2/intersubject_generalization/

# Run the job
python linear_intersubject_generalization_utils.py -inp $inp_dir -out $out_dir -method $method -dim_reduction $dim_reduction -n_comp $n_components 
