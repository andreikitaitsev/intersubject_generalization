#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=learn__pr_incr
#SBATCH --mail-type=end
#SBATCH --mem=30000
#SBATCH --time=30:00:00
#SBATCH --qos=prio

# run intersubject generalization on 100 random shuffles of train data 
# percent of training images to use: 5, 10, 20, 40, 80, 100

# N JOBS = 600

step_list=(5 10 20 40 80 100)
nsplits=100
nshuffles=100

method="multiviewica"
prepr="pca"
n_comps=200
inp_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/featurematrices/50hz/time_window13-40/"
out_dir_base="/scratch/akitaitsev/intersubject_generalizeation/linear/intersubject_generalization/dataset1/learn_projections_incrementally/shuffle_splits/100splits/100shuffles/"$method"/"$prepr"/"$n_comps"/"

declare -a seeds
declare -a shuffles
declare -a out_dirs
declare -a steps

ind=0
seed_cntr=0
for shuffle in $(seq 0 $((nshuffles-1)))
do
    for step in ${step_list[@]}
    do
        out_dirs[$ind]=$out_dir_base"/50hz/time_window13-40/shuffle_"$shuffle"/step_"$step"/"
        shuffles[$ind]=$shuffle
        seeds[$ind]=$seed_cntr
        steps[$ind]=$step
        ((seed_cntr+=1))
        ((ind=ind+1))
    done
done

seeds=${seeds[$SLURM_ARRAY_TASK_ID]}
steps=${steps[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
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
