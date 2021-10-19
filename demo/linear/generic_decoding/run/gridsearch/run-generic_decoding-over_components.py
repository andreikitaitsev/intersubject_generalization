#! /bin/bash
#SBATCH --job-name=gen_dec_comp
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=00:10:00
#SBATCH --qos=prio

# 16 jobs
out_base="/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/"
real_base="/scratch/akitaitsev/intersubject_generalizeation/linear/intersubject_generalization/"
pred_base="/scratch/akitaitsev/intersubject_generalizeation/linear/regression/"
method="multiviewica"

declare -a out_dir
declare -a real_files
declare -a pred_files
declare -a dtypes

ind=0
for prepr in "pca" "srm"
do
    for n_comp in 10 50 200 400
    do
        for dtype in "average" "subjectwise"
        do
            out_dir[$ind]=$out_base$method"/main/"$prepr"/"$n_comp"/50hz/time_window13-40/"
            real_files[$ind]=$real_base$method"/main/"$prepr"/"$n_comp"/50hz/time_window13-40/shared_test.pkl"
            pred_files[$ind]=$pred_base$method"/main/"$prepr"/"$n_comp"/50hz/time_window13-40/Y_test_predicted_"$dtype".pkl"
            dtypes[$ind]=$dtype
            ((ind=ind+1))
        done
    done
done

out_dir=${out_dir[$SLURM_ARRAY_TASK_ID]}
real_files=${real_files[$SLURM_ARRAY_TASK_ID]}
pred_files=${pred_files[$SLURM_ARRAY_TASK_ID]}
dtypes=${dtypes[$SLURM_ARRAY_TASK_ID]}

echo SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID
echo dtype: $dtypes
echo out_dir: $out_dir
echo pred_file: $pred_files
echo real_file: $real_files

cd  /home/akitaitsev/code/intersubject_generalization/linear/generic_decoding/
python generic_decoding.py -real $real_files -pred $pred_files -d_type $dtypes -out $out_dir 
