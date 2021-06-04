#! /bin/bash    
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=permica_pca200
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=70:00:00
#SBATCH --qos=prio

### Run multiviewICA with srm=200 on train data and use leared matrices to project 
### and backproject val and test data'''

# N JOBS = 2

# 100 hz version
t_winds=("0-80" "26-80")
srate=100
inp_dir="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/featurematrices/"$srate"hz/time_window"
out_dir="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/intersubject_generalization/permica/pca_200/"$srate"hz/time_window"

declare -a inp_dirs
declare -a out_dirs
ind=0
for t in ${t_winds[@]}
do
    inp_dirs[$ind]=$inp_dir$t"/av_reps/"
    out_dirs[$ind]=$out_dir$t"/av_reps/"
    ((ind=ind+1))
done

method="permica"
dim_reduction="pca"
n_components="200"

echo "Running permica"
sleep 10

### Extracting the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
inp_dirs=${inp_dirs[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset2/intersubject_generalization/

# Run the job
python linear_intersubject_generalization_utils.py -inp $inp_dirs -out $out_dirs -method $method -dim_reduction $dim_reduction -n_comp $n_components
