#! /bin/bash    
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=mvica_srm200
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=24:00:00
#SBATCH --qos=prio

### Run multiviewICA with srm=200 on train data and use leared matrices to project 
### and backproject val and test data'''

inp_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/featurematrices/time_window"
out_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/intersubject_generalization/multiviewica/main/srm/200/time_window"

declare -a inp_dirs
declare -a out_dirs
ind=0
for t in "0-40" "13-40"
do
    inp_dirs[$ind]=$inp_dir$t
    out_dirs[$ind]=$out_dir$t
    ((ind=ind+1))
done

method="multiviewica"
dim_reduction="srm"
n_components="200"

echo "Running multiviewica"
sleep 10

### Extracting the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
inp_dirs=${inp_dirs[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}

cd /home/akitaitsev/code/intersubject_generalization/linear/intersubject_generalization/

# Run the job
python linear_intersubject_generalization_utils.py -inp $inp_dirs -out $out_dirs -method $method -dim_reduction $dim_reduction -n_comp $n_components
