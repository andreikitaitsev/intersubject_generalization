#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=LR_sat_prof
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=00:20:00
#SBATCH --qos=prio

# Dataset 1
# run linear regression for every step of every shuffle for
# AVERAGE data

# steps: 10 20 30 40 50 60 70 80 90 100

# N JOBS = 300

step_list=(10 20 30 40 50 60 70 80 90 100)
nsplits=10
nshuffles=10
method_list=("multiviewica" "groupica" "permica")
prepr="pca"
n_comps=200
regr_type="average"

dnn_dir="/scratch/akitaitsev/encoding_Ale/dataset1/dnn_activations/"
eeg_dir_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/intersubject_generalization/"\
"saturation_profile-different_methods/"
out_dir_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/regression/saturation_profile-different_methods/"

declare -a eeg_dirs
declare -a out_dirs
declare -a shuffles
declare -a steps

ind=0
for shuffle in $(seq 0 $((nshuffles-1)))
do
    for step in ${step_list[@]}
    do
        for method in ${method_list[@]}
        do
        eeg_dirs[$ind]=$eeg_dir_base"/"$method"/50hz/shuffle_"$shuffle"/step_"$(printf '%d' $step)"/"
        out_dirs[$ind]=$out_dir_base"/"$method"/50hz/shuffle_"$shuffle"/step_"$(printf '%d' $step)"/"
        shffles[$ind]=$shuffle
        steps[$ind]=$step
        ((ind=ind+1))
        done
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
echo regr_type: $regr_type

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset1/regression
python linear_regression.py -eeg_dir $eeg_dirs -dnn_dir $dnn_dir -out_dir $out_dirs -regr_type $regr_type -learn_pr_incr
