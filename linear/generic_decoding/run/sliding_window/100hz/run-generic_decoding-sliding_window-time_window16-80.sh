#! /bin/bash
#SBATCH --job-name=gd_sl_wind
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=00:10:00
#SBATCH --qos=prio

# 8 jobs
out_base="/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/sliding_window/multiviewica/main/"
real_base="/scratch/akitaitsev/intersubject_generalizeation/linear/intersubject_generalization/sliding_window/multiviewica/main/"
pred_base="/scratch/akitaitsev/intersubject_generalizeation/linear/regression/sliding_window/multiviewica/main/"

declare -a out_dir
declare -a real_files
declare -a pred_files
declare -a dtypes

ind=0
for prepr in "pca" "srm"
do
    for n_comp in 10 50 
    do
        for dtype in "average" "subjectwise"
        do
            out_dir[$ind]=$out_base$prepr"/"$n_comp"/100hz/time_window16-80/"
            real_files[$ind]=$real_base$prepr"/"$n_comp"/100hz/time_window16-80/shared_test.pkl"
            pred_files[$ind]=$pred_base$prepr"/"$n_comp"/100hz/time_window16-80/Y_test_predicted_"$dtype".pkl"
            dtypes[$ind]=$dtype
            ((ind=ind+1))
        
        done
    done    
done

out_dir=${out_dir[$SLURM_ARRAY_TASK_ID]}
real_files=${real_files[$SLURM_ARRAY_TASK_ID]}
pred_files=${pred_files[$SLURM_ARRAY_TASK_ID]}
dtypes=${dtypes[$SLURM_ARRAY_TASK_ID]}

echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
echo out_dir: $out_dir
echo pred_file: $pred_files
echo real_file: $real_files
echo dtype: $dtypes

cd  /home/akitaitsev/code/intersubject_generalization/linear/generic_decoding/
python generic_decoding.py -real $real_files -pred $pred_files -d_type $dtypes -out $out_dir -sliding_window
