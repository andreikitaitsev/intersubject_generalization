#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=CV_IGA
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=30:00:00
#SBATCH --qos=prio

# Run CV intersubject generalization 
# N JOBS=4
# function parameters
method_list=("multiviewica" "groupica" "permica" "control")
seed=0
inp_dir="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/featurematrices/100hz/time_window26-80/"
out_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/intersubject_generalization/cv/"

declare -a out_dirs
declare -a methods
ind=0
for method in ${method_list[@]}
do
    methods[$ind]=$method
    out_dirs[$ind]=$out_base$method"/100hz/time_window26-80/"
    ((ind=ind+1))
done

out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
methods=${methods[$SLURM_ARRAY_TASK_ID]}

echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "method:" $methods
echo "output_dir: " $out_dirs

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset1/intersubject_generalization/

python arbitrary_train_data_ratio.py -inp $inp_dir -out $out_dirs -method $methods -dim_reduction pca -n_comp 200 -seed $seed -ratio 10
