#! /bin/bash
#SBATCH --job-name=gen_dec
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=00:10:00
#SBATCH --qos=prio


# run generic decoding on data predicted with leave one out cross validation. Note, 
# CV in regression, not in intersubject generalization step!

# N JOBS = 6

out_base="/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/cross_val_regr/"
real_base="/scratch/akitaitsev/intersubject_generalizeation/linear/intersubject_generalization/"
pred_base="/scratch/akitaitsev/intersubject_generalizeation/linear/regression/cross_val_regr/"

n_splits=7
n_comp=200
declare -a out_dirs
declare -a real_files
declare -a pred_files

ind=0
for method in "multiviewica" "groupica" "permica"
do 
    for prepr in "pca" "srm"
    do
        real_files[$ind]=$real_base"/"$method"/main/"$prepr"/"$n_comp"/50hz/time_window13-40/shared_test.pkl"        
        pred_files[$ind]=$pred_base"/"$method"/"$prepr"/"$n_comp"/50hz/time_window13-40/Y_test_predicted_.pkl"
        out_dirs[$ind]=$out_base"/"$method"/"$prepr"/"$n_comp"/50hz/time_window13-40/"
        ((ind=ind+1))
    done
done

real_files=${real_files[$SLURM_ARRAY_TASK_ID]}
pred_files=${pred_files[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}

echo Running linear regression with leave one out cross-validation.
echo SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID
echo real_file: $real_files
echo pred_file: $pred_files
echo out_dir: $out_dirs

cd /home/akitaitsev/code/intersubject_generalization/linear/generic_decoding/
python generic_decoding.py -real $real_files -pred $pred_files -out $out_dirs -cv_regr
