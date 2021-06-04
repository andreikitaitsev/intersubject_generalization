#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=regr
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=00:05:00
#SBATCH --qos=prio

# N JOBS = 60
nsteps=10
dnn_dir="/scratch/akitaitsev/encoding_Ale/dnn_activations/dataset1/"
eeg_dir_base="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/intersubject_generalization/learn_projections_incrementally/"
out_dir_base="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/regression/learn_projections_incrementally/"
method="multiviewica"
prepr="pca"

declare -a eeg_dirs
declare -a out_dirs
declare -a regr_types
declare -a n_comps
declare -a steps
ind=0
for n_comp in "50" "200" "400"
do
    for step in $(seq 0 $((nsteps-1)))
    do
        for regr_type in "average" "subjectwise"
        do
            eeg_dirs[$ind]=$eeg_dir_base"/"$method"/"$prepr"/"$n_comp"/50hz/time_window13-40/step_"$(printf '%d' $step)"/"
            out_dirs[$ind]=$out_dir_base"/"$method"/"$prepr"/"$n_comp"/50hz/time_window13-40/step_"$(printf '%d' $step)"/"
            regr_types[$ind]=$regr_type
            n_comps[$ind]=$n_comp
            steps[$ind]=$step
            ((ind=ind+1))
        done
    done
done

eeg_dirs=${eeg_dirs[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
regr_types=${regr_types[$SLURM_ARRAY_TASK_ID]}
n_comps=${n_comps[$SLURM_ARRAY_TASK_ID]}
steps=${steps[$SLURM_ARRAY_TASK_ID]}

echo SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID
echo eeg_dir: $eeg_dirs
echo out_dir: $out_dirs
echo step: $steps
echo n_comp: $n_comps
echo regr_type: $regr_types

cd /home/akitaitsev/code/intersubject_generalization/linear/regression
python linear_regression.py -eeg_dir $eeg_dirs -dnn_dir $dnn_dir -out_dir $out_dirs -regr_type $regr_types -learn_pr_incr
