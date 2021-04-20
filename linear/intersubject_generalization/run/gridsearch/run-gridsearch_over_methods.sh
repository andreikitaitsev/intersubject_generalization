#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=grids_200
#SBATCH --mail-type=end
#SBATCH --mem=20000
#SBATCH --time=24:00:00
#SBATCH --qos=prio

### Run gridsearch over differnet intersubject generalization algorithms
### with 200 components for time window 13-40

# 6 task arrays
inp_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/featurematrices/time_window13-40/"
out_dir_base="/scratch/akitaitsev/intersubject_generalizeation/linear/intersubject_generalization/" 

declare -a methods 
declare -a prepr
declare -a out_dirs

ind=0
for method in "multiviewica" "permica" "groupica"
do
    for prepr in "pca" "srm"
    do
        out_dirs[$ind]=$out_dir_base$method"/main/"$prepr"/200/time_window13-40/"
        methods[$ind]=$method
        preprocessors[$ind]=$prepr
        ((ind=ind+1))
    done
done

sleep 10

### Extracting the parameters
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
methods=${methods[$SLURM_ARRAY_TASK_ID]}
preprocessors=${preprocessors[$SLURM_ARRAY_TASK_ID]}

echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "method: " $methods
echo "preprocessor: " $preprocessors

cd /home/akitaitsev/code/intersubject_generalization/linear/intersubject_generalization/
python linear_intersubject_generalization_utils.py -inp $inp_dir -out $out_dirs -method $methods -dim_reduction $preprocessors -n_comp 200
