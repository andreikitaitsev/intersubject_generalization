#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=regr
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=00:05:00
#SBATCH --qos=prio

# N JOBS = 8

n_splits=7 # leave one subject out

dnn_dir="/scratch/akitaitsev/encoding_Ale/dnn_activations/dataset1/"
eeg_dir_base="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/intersubject_generalization/"
out_dir_base="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/regression/"

declare -a eeg_dirs
declare -a out_dirs
declare -a methods
declare -a n_comps 
declare -a preprs
declare -a methods

n_comp=200
ind=0
for method in "control" "multiviewica" "permica" "groupica"
do
    for prepr in "pca" "srm"
    do
        eeg_dirs[$ind]=$eeg_dir_base"/"$method"/main/"$prepr"/"$n_comp"/50hz/time_window13-40/"
        out_dirs[$ind]=$out_dir_base"cross_val_regr/"$method"/"$prepr"/"$n_comp"/50hz/time_window13-40/"
        methods[$ind]=$method
        preprs[$ind]=$prepr
        n_comps[$ind]=$n_comp
        ((ind=ind+1))
    done
done

eeg_dirs=${eeg_dirs[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
preprs=${preprs[$SLURM_ARRAY_TASK_ID]}
methods=${methods[$SLURM_ARRAY_TASK_ID]}

echo Running linear regression with cross validation over subejcts 
echo eeg dir: $eeg_dirs
echo out dir: $out_dirs
echo method: $methods
echo prepr: $preprs

cd /home/akitaitsev/code/intersubject_generalization/linear/regression/
python linear_regression.py -eeg_dir $eeg_dirs -dnn_dir $dnn_dir -out_dir $out_dirs -n_splits $n_splits -cv_regr
