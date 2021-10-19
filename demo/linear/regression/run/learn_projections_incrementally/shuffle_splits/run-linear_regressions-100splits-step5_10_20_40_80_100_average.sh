#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=learn__pr_incr
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=00:20:00
#SBATCH --qos=prio

# run linear regression for every step of every shuffle
# for average data

# steps: 5 10 20 40 80 100

# N JOBS = 600 

step_list=(5 10 20 40 80 100)
nsplits=100
nshuffles=100
method="multiviewica"
prepr="pca"
n_comps=200
regr_type="average"

dnn_dir="/scratch/akitaitsev/encoding_Ale/dnn_activations/dataset1/"
eeg_dir_base="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/intersubject_generalization/"\
"learn_projections_incrementally/shuffle_splits/"$nsplits"splits/"$nshuffles"shuffles/"$method"/"$prepr"/"$n_comps"/"
out_dir_base="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/regression/learn_projections_incrementally/"\
"shuffle_splits/"$nsplits"splits/"$nshuffles"shuffles/"$method"/"$prepr"/"$n_comps"/"

declare -a eeg_dirs
declare -a out_dirs
declare -a shuffles
declare -a steps

ind=0
for shuffle in $(seq 0 $((nshuffles-1)))
do
    for step in ${step_list[@]}
    do
    eeg_dirs[$ind]=$eeg_dir_base"/50hz/time_window13-40/shuffle_"$shuffle"/step_"$(printf '%d' $step)"/"
    out_dirs[$ind]=$out_dir_base"/50hz/time_window13-40/shuffle_"$shuffle"/step_"$(printf '%d' $step)"/"
    shffles[$ind]=$shuffle
    steps[$ind]=$step
    ((ind=ind+1))
    done
done
                                                                                                                                
eeg_dirs=${eeg_dirs[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
steps=${steps[$SLURM_ARRAY_TASK_ID]}
shuffles=${shuffles[$SLURM_ARRAY_TASK_ID]}

echo SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID
echo eeg_dir: $eeg_dirs
echo out_dir: $out_dirs
echo step: $steps
echo n_comp: $n_comps
echo regr_type: $regr_type

cd /home/akitaitsev/code/intersubject_generalization/linear/regression
python linear_regression.py -eeg_dir $eeg_dirs -dnn_dir $dnn_dir -out_dir $out_dirs -regr_type $regr_type -learn_pr_incr
