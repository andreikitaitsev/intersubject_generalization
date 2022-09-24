#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=sl_win_regr
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=01:00:00
#SBATCH --qos=prio

# N_JOBS = 8
# Run separate linear regression on each sliding window 
dnn_dir="/scratch/akitaitsev/encoding_Ale/dnn_activations/dataset1/"
eeg_dir_base="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/intersubject_generalization/sliding_window/multiviewica/main/"
out_dir_base="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/regression/sliding_window/multiviewica/main/"

declare -a eeg_dirs
declare -a out_dirs
declare -a regr_types
declare -a n_comps

ind=0
for prepr in "pca" "srm"
do
    for n_comp in 10 50
    do
        for regr_type in "average" "subjectwise"
        do
            regr_types[$ind]=$regr_type
            n_comps[$ind]=$n_comp
            eeg_dirs[$ind]=$eeg_dir_base"/"$prepr"/"$n_comp"/100hz/time_window26-80/"
            out_dirs[$ind]=$out_dir_base"/"$prepr"/"$n_comp"/100hz/time_window26-80/"
            ((ind=ind+1))
        done
    done
done

eeg_dirs=${eeg_dirs[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
regr_types=${regr_types[$SLURM_ARRAY_TASK_ID]}
n_comps=${n_comps[$SLURM_ARRAY_TASK_ID]}

echo SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID
echo eeg_dir: $eeg_dirs
echo out_dir: $out_dirs
echo n_components: $n_comps
echo regr_type: $regr_types

cd /home/akitaitsev/code/intersubject_generalization/linear/regression/
python linear_regression.py -dnn_dir $dnn_dir -eeg_dir $eeg_dirs -out_dir $out_dirs -regr_type $regr_types -sliding_window 
