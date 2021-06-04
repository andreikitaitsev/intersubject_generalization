#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=sl_wind_mvica
#SBATCH --mail-type=end
#SBATCH --mem=20000
#SBATCH --time=30:00:00
#SBATCH --qos=prio

# Run sliding window intersubject generalization for the best methods (mvica with pca and srm)
# N JOBS=4
# function parameters
method="multiviewica"
wind_len=5

inp_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset_matrices/100hz/time_window16-80/"
out_base="/scratch/akitaitsev/intersubject_generalizeation/linear/intersubject_generalization/sliding_window/multiviewica/main/"

declare -a out_dirs
declare -a preprocessors
declare -a n_comps
ind=0
for prepr in "pca" "srm"
do
    for n_comp in 10 50
    do
        n_comps[$ind]=$n_comp
        out_dirs[$ind]=$out_base$prepr"/"$n_comp"/100hz/time_window16-80/"
        preprocessors[$ind]=$prepr
        ((ind=ind+1))
    done
done

out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
preprocessors=${preprocessors[$SLURM_ARRAY_TASK_ID]}
n_comps=${n_comps[$SLURM_ARRAY_TASK_ID]}

echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "preprocessor: " $preprocessors
echo "n_comp: " $n_comps
echo "output_dir: " $out_dirs

cd /home/akitaitsev/code/intersubject_generalization/linear/intersubject_generalization/

python sliding_window.py -inp $inp_dir -out $out_dirs -method $method -dim_reduction $preprocessors -n_comp $n_comp -wind_len $wind_len
