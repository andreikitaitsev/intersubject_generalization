#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=sl_win_regr
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=01:00:00
#SBATCH --qos=prio

# N_JOBS = 4
# Run separate linear regression with CV on controls
dnn_dir="/scratch/akitaitsev/encoding_Ale/dataset2/dnn_activations/"
eeg_dir="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/featurematrices/100hz/time_window26-80/av_reps/"
out_dir_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/regression/cv/"
seed=0

ratio_list=(10 100)
declare -a out_dirs
declare -a ratios
declare -a cvs

ind=0
for ratio in ${ratio_list[@]}
do
    for cv in 0 1
    do
        out_dirs[$ind]=$out_dir_base"/control"$ratio"-cv"$cv"/100hz/time_window26-80/"
        ratios[$ind]=$ratio
        cvs[$ind]=$cv
        ((ind=ind+1))
    done
done

out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
ratios=${ratios[$SLURM_ARRAY_TASK_ID]}
cvs=${cvs[$SLURM_ARRAY_TASK_ID]}

echo SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID
echo eeg_dir: $eeg_dir
echo out_dir: $out_dirs
echo ratio: $ratios

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset2/regression/
python linear_regression-cv.py -dnn_dir $dnn_dir -eeg_dir $eeg_dir -out_dir $out_dirs -is_raw -cv $cvs -ratio $ratios -seed $seed
