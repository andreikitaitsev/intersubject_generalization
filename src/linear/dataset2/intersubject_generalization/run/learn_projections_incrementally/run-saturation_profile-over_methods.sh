#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=learn__pr_incr
#SBATCH --mail-type=end
#SBATCH --mem=30000
#SBATCH --time=30:00:00
#SBATCH --qos=prio

# run intersubject generalization on 10 random shuffles of train data 
# with the steps of 10 
# for MULTIVEIWICA PERMICA GROUPICA

# N JOBS = 300

step_list=(10 20 30 40 50 60 70 80 90 100)
method_list=("multiviewica" "permica" "groupica")
nsplits=100
nshuffles=10

prepr="pca"
n_comps=200
inp_dir="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/featurematrices/50hz/time_window13-40/av_reps/"
out_dir_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/intersubject_generalization/saturation_profile-different_methods/"

# print out all the counters
check=0

declare -a methods
declare -a seeds
declare -a shuffles
declare -a out_dirs
declare -a steps

ind=0
for method in ${method_list[@]}
do
    for step in ${step_list[@]}
    do
        seed_cntr=0
        for shuffle in $(seq 0 $((nshuffles-1)))
        do
            out_dirs[$ind]=$out_dir_base"/"$method"/50hz/shuffle_"$shuffle"/step_"$step"/"
            methods[$ind]=$method
            shuffles[$ind]=$shuffle
            seeds[$ind]=$seed_cntr
            steps[$ind]=$step
            ((seed_cntr+=1))
            ((ind=ind+1))
            if [[ $check == 1 ]]
                then 
                echo ""
                echo "shuffle" $shuffle
                echo $method
                echo "seed" $seed_cntr
                echo "step" $step
            fi
        done
    done
done

shuffles=${shuffles[$SLURM_ARRAY_TASK_ID]}
methods=${methods[$SLURM_ARRAY_TASK_ID]}
seeds=${seeds[$SLURM_ARRAY_TASK_ID]}
steps=${steps[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
steps=${steps[$SLURM_ARRAY_TASK_ID]}

echo SLURM ARRAY TASK ID: $SLURM_ARRAY_TASK_ID
echo method: $methods
echo shuffle: $shuffles
echo step: $steps
echo seed: $seeds
echo output_dir: $out_dirs

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset2/intersubject_generalization
python learn_projections_incrementally_shuffle_splits.py -inp $inp_dir -out $out_dirs -method $methods -dim_reduction "pca" -n_comp 200 -nsplits $nsplits -step $steps -seed $seeds -skip_val
