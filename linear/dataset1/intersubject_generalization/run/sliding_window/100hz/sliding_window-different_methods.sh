#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=sl_wind_mvica
#SBATCH --mail-type=end
#SBATCH --mem=20000
#SBATCH --time=30:00:00
#SBATCH --qos=prio

# Run sliding window intersubject generalization for the best methods (mvica with pca and srm)
# N JOBS=3
# function parameters
method_list=("multiviewica" "groupica" "permica")
wind_len=5

inp_dir="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/dataset_matrices/100hz/time_window16-80/"
out_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/intersubject_generalization/sliding_window-different_emthods/"

declare -a out_dirs
declare -a methods
ind=0
for method in ${method_list[@]}
do
    methods[$ind]=$method
    out_dirs[$ind]=$out_base$method"/100hz/"
    ((ind=ind+1))
done

out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
methods=${methods[$SLURM_ARRAY_TASK_ID]}

echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "method:" $methods
echo "output_dir: " $out_dirs

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset1/intersubject_generalization/

python sliding_window.py -inp $inp_dir -out $out_dirs -method $method -dim_reduction pca -n_comp 200 -wind_len $wind_len
