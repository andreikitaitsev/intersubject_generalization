#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=sl_wind_mvica
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=24:00:00
#SBATCH --qos=prio

# Run sliding window intersubject generalization for the best methods (mvica with pca and srm)
# N JOBS=2
# function parameters
method="multiviewica"
n_comp=50
wind_len=5

inp_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset_matrices/50hz/time_window13-40/"
out_base="/scratch/akitaitsev/intersubject_generalizeation/linear/intersubject_generalization/sliding_window/multiviewica/main/"

declare -a out_dirs
declare -a preprocessors
ind=0
for prepr in "pca" "srm"
do
    out_dirs[$ind]=$out_base$prepr"/200/time_window13_40/"
    preprocessors[$ind]=$prepr
    ((ind=ind+1))
done

out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
preprocessors=${preprocessors[$SLURM_ARRAY_TASK_ID]}

echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "output_dir: " $out_dirs
echo "preprocessor: " $preprocessors

cd /home/akitaitsev/code/intersubject_generalization/linear/intersubject_generalization/
python sliding_window.py -inp $inp_dir -out $out_dirs -method $method -dim_reduction $preprocessors -n_comp $n_comp -wind_len $wind_len


