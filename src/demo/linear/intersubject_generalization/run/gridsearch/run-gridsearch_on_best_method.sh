#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=grid_best_meth
#SBATCH --mail-type=end
#SBATCH --mem=20000
#SBATCH --time=24:00:00
#SBATCH --qos=prio

# Run gridsearch on 2 best intersubject generalization methods to determine the 
# optimal number of components in pca in srm preprocessing.
# N_JOBS = 10

inp_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/featurematrices/time_window13-40/"
out_dir_base="/scratch/akitaitsev/intersubject_generalizeation/linear/intersubject_generalization/"
method="multiviewica"

declare -a preprocessors 
declare -a n_components
declare -a out_dirs

ind=0
for prepr in "pca" "srm"
do
    for n_comp in "None" 10 50 200 400
    do 
        preprocessors[$ind]=$prepr
        n_components[$ind]=$n_comp
        out_dirs[$ind]=$out_dir_base$method"/main/"$prepr"/"$n_comp"/time_window13-40/"
        ((ind=ind+1))
    done
done

sleep 10

### Extracting the parameters
preprocessors=${preprocessors[$SLURM_ARRAY_TASK_ID]}
n_components=${n_components[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}

echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "preprocessor: " $preprocessors
echo "n components: " $n_components
echo "output directory: " $out_dirs 

cd /home/akitaitsev/code/intersubject_generalization/linear/intersubject_generalization/
python linear_intersubject_generalization_utils.py -inp $inp_dir -out $out_dirs -method $method -dim_reduction $preprocessors -n_comp $n_components
