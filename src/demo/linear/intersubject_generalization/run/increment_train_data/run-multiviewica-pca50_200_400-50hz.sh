#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=incr
#SBATCH --mail-type=end
#SBATCH --mem=20000
#SBATCH --time=20:00:00
#SBATCH --qos=prio

# Run intersubject generalization for multiviewica with different number of
# components for each of 10 bins of incremental training featurematrices

# N JOBS = 30

nsteps=10
inp_base="/scratch/akitaitsev/intersubject_generalizeation/linear/featurematrices/incremental/50hz/time_window13-40/"
out_base="/scratch/akitaitsev/intersubject_generalizeation/linear/intersubject_generalization/incremental/"
method="multiviewica"
prepr="pca"

declare -a inp_dirs
declare -a out_dirs
declare -a steps
declare -a n_comps

ind=0
for n_comp in 50 200 400
do
    for step in $(seq 0 $((nsteps-1)))
    do
        inp_dirs[$ind]=$inp_base"/step_"$(printf '%d' $step)"/"
        out_dirs[$ind]=$out_base"/"$method"/"$prepr"/"$n_comp"/50hz/time_window13-40/step_"$(printf '%d' $step)"/"
        steps[$ind]=$(printf '%d' $step)
        n_comps[$ind]=$n_comp
        ((ind=$ind+1))
    done
done

inp_dirs=${inp_dirs[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
steps=${steps[$SLURM_ARRAY_TASK_ID]}
n_comps=${n_comps[$SLURM_ARRAY_TASK_ID]}

echo SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID
echo inp_dir: $inp_dirs
echo out_dir: $out_dirs
echo n_comp: $n_comps
echo step: $steps

cd /home/akitaitsev/code/intersubject_generalization/linear/intersubject_generalization
python linear_intersubject_generalization_utils.py -inp $inp_dirs -out $out_dirs -method $method -dim_reduction $prepr -n_comp $n_comps
