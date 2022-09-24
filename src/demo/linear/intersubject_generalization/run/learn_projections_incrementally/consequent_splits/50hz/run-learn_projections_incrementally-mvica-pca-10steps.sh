#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=learn__pr_incr
#SBATCH --mail-type=end
#SBATCH --mem=20000
#SBATCH --time=24:00:00
#SBATCH --qos=prio

# run learn_projections_incrementally for mvica with pca 50, 200 and 400 comps
# for 50 hz data for time window 13-40 samples

# N JOBS = 30
method="multiviewica"
prepr="pca"
nsteps=10
inp_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/featurematrices/50hz/time_window13-40/"
out_dir_base="/scratch/akitaitsev/intersubject_generalizeation/linear/intersubject_generalization/learn_projections_incrementally/multiviewica/pca/"

declare -a n_comps
declare -a out_dirs
declare -a splits

ind=0
for n_comp in 50 200 400
do
    for split in $(seq 0 $((nsteps-1)))
    do
        splits[$ind]=$split
        out_dirs[$ind]=$out_dir_base"/"$n_comp"/50hz/time_window13-40/step_"$split"/"
        n_comps[$ind]=$n_comp
        ((ind=ind+1))
    done
done

splits=${splits[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
n_comps=${n_comps[$SLURM_ARRAY_TASK_ID]}

echo SLURM ARRAY TASK ID: $SLURM_ARRAY_TASK_ID
echo method: $method
echo preproessing: $prepr
echo n_components: $n_comps
echo output_dir: $out_dirs
echo split: $splits

cd /home/akitaitsev/code/intersubject_generalization/linear/intersubject_generalization
python learn_projections_incrementally.py -inp $inp_dir -out $out_dirs -method $method -dim_reduction $prepr -n_comp $n_comps -nsteps $nsteps -split2use $splits 
