#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=learn__pr_incr
#SBATCH --mail-type=end
#SBATCH --mem=40000
#SBATCH --time=30:00:00
#SBATCH --qos=prio

# run intersubject generalization on 10 random shuffles of train data between 1 and 10%

# N JOBS = 300

nsplits=100
nshuffles=10

method="multiviewica"
prepr="pca"
inp_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/featurematrices/50hz/time_window13-40/"
out_dir_base="/scratch/akitaitsev/intersubject_generalizeation/linear/intersubject_generalization/learn_projections_incrementally/shuffle_splits/100splits/"$method"/"$prepr

declare -a steps
declare -a seeds
declare -a shuffles
declare -a out_dirs
declare -a n_comps

ind=0
for n_comp in 50 200 400
do
    seed_cntr=0
    for shuffle in $(seq 0 $((nshuffles-1)))
    do
        for step in $(seq 0 9)
        do
            out_dirs[$ind]=$out_dir_base"/"$n_comp"/50hz/time_window13-40/shuffle_"$shuffle"/step_"$step"/"
            n_comps[$ind]=$n_comp
            shuffles[$ind]=$shuffle
            seeds[$ind]=$seed_cntr
            steps[$ind]=$step
            ((seed_cntr+=1))
            ((ind=ind+1))
        done
    done
done

seeds=${seeds[$SLURM_ARRAY_TASK_ID]}
steps=${steps[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
n_comps=${n_comps[$SLURM_ARRAY_TASK_ID]}
steps=${steps[$SLURM_ARRAY_TASK_ID]}

echo SLURM ARRAY TASK ID: $SLURM_ARRAY_TASK_ID
echo method: $method
echo preproessing: $prepr
echo n_components: $n_comps
echo shuffle: $shuffles
echo step: $steps
echo seed: $seeds
echo output_dir: $out_dirs

cd /home/akitaitsev/code/intersubject_generalization/linear/intersubject_generalization
python learn_projections_incrementally_shuffle_splits.py -inp $inp_dir -out $out_dirs -method $method -dim_reduction $prepr -n_comp $n_comps -nsplits $nsplits -step $steps -seed $seeds
