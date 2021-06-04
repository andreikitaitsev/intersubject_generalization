#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=learn__pr_incr
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=00:10:00
#SBATCH --qos=prio

# run linear regression for every step of every shuffle
# N JOBS = 600 

nsplits=100
nsteps=10
nshuffles=10
method="multiviewica"
prepr="pca"
dnn_dir="/scratch/akitaitsev/encoding_Ale/dnn_activations/dataset1/"
eeg_dir_base="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/intersubject_generalization/learn_projections_incrementally/shuffle_splits/"$nsplits"splits/"$method"/"$prepr
out_dir_base="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/regression/learn_projections_incrementally/shuffle_splits/"$nsplits"splits/"$method"/"$prepr

declare -a eeg_dirs
declare -a out_dirs
declare -a regr_types
declare -a n_comps
declare -a steps
declare -a shuffles

ind=0
for n_comp in 50 200 400
do
    for shuffle in $(seq 0 $((nshuffles-1)))
    do
        for step in $(seq 0 $((nsteps-1)))
        do
            for regr_type in "average" "subjectwise"
            do
            eeg_dirs[$ind]=$eeg_dir_base"/"$n_comp"/50hz/time_window13-40/shuffle_"$shuffle"/step_"$(printf '%d' $step)"/"
            out_dirs[$ind]=$out_dir_base"/"$n_comp"/50hz/time_window13-40/shuffle_"$shuffle"/step_"$(printf '%d' $step)"/"
            regr_types[$ind]=$regr_type
            shffles[$ind]=$shuffle
            n_comps[$ind]=$n_comp
            steps[$ind]=$step
            ((ind=ind+1))
            done
        done
    done
done
                                                                                                                                
eeg_dirs=${eeg_dirs[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
regr_types=${regr_types[$SLURM_ARRAY_TASK_ID]}
n_comps=${n_comps[$SLURM_ARRAY_TASK_ID]}
steps=${steps[$SLURM_ARRAY_TASK_ID]}
shuffles=${shuffles[$SLURM_ARRAY_TASK_ID]}

echo SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID
echo eeg_dir: $eeg_dirs
echo out_dir: $out_dirs
echo step: $steps
echo n_comp: $n_comps
echo regr_type: $regr_types

cd /home/akitaitsev/code/intersubject_generalization/linear/regression
python linear_regression.py -eeg_dir $eeg_dirs -dnn_dir $dnn_dir -out_dir $out_dirs -regr_type $regr_types -learn_pr_incr
