#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=regr
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=00:05:00
#SBATCH --qos=prio

# N JOBS = 9

dnn_dir="/scratch/akitaitsev/encoding_Ale/dnn_activations/dataset1/"
eeg_dir_base="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/intersubject_generalization/cross_val/"
out_dir_base="/scratch/akitaitsev/intersubject_generalizeation/linear/dataset1/regression/"
prepr="pca"

declare -a eeg_dirs
declare -a out_dirs
declare -a methods
declare -a n_comps 
declare -a n_comps

ind=0
for method in "multiviewica" "permica" "groupica"
do
    for n_comp in 50 200 400
    do
        eeg_dirs[$ind]=$eeg_dir_base"/"$method"/main/"$prepr"/"$n_comp"/50hz/time_window13-40/"
        out_dirs[$ind]=$out_dir_base"cv_gener/"$method"/"$prepr"/"$n_comp"/50hz/time_window13-40/"
        methods[$ind]=$method
        preprs[$ind]=$prepr
        n_comps[$ind]=$n_comp
        ((ind=ind+1))
    done
done

eeg_dirs=${eeg_dirs[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
methods=${methods[$SLURM_ARRAY_TASK_ID]}
n_comps=${n_comps[$SLURM_ARRAY_TASK_ID]}

echo Running linear regression with cross validation over subejcts 
echo eeg dir: $eeg_dirs
echo out dir: $out_dirs
echo method: $methods
echo prepr: $prepr
echo n_comps: $n_comps

cd /home/akitaitsev/code/intersubject_generalization/linear/regression/
python linear_regression.py -eeg_dir $eeg_dirs -dnn_dir $dnn_dir -out_dir $out_dirs -cv_gener 
