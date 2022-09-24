#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=cv_par
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=04:00:00
#SBATCH --qos=prio

# N JOBS = 9

n_splits=7
prepr="pca"
inp_base="/scratch/akitaitsev/intersubject_generalizeation/linear/intersubject_generalization/cross_val/"
declare -a inp_dirs

ind=0
for method in "multiviewica" "permica" "groupica"
do
    for n_comp in 50 200 400
    do
        inp_dirs[$ind]=$inp_base"/"$method"/main/"$prepr"/"$n_comp"/50hz/time_window13-40/"
        ((ind=ind+1))
    done
done

inp_dirs=${inp_dirs[$SLURM_ARRAY_TASK_ID]}
echo SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID
echo inp_dir/out_dir: $inp_dirs

cd /home/akitaitsev/code/intersubject_generalization/linear/intersubject_generalization

python cross_val_single_split-merge_splits.py -inp $inp_dirs -out $inp_dirs -n_splits $n_splits
