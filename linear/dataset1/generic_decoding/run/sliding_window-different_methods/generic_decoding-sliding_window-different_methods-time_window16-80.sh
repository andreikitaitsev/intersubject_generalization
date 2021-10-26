#! /bin/bash
#SBATCH --job-name=gd_sl_wind
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=00:10:00
#SBATCH --qos=prio

# 8 jobs
out_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/generic_decoding/sliding_window-different_methods/"
real_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/intersubject_generalization/sliding_window-different_methods/"
pred_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/regression/sliding_window-different_methods/"

declare -a out_dirs
declare -a real_files
declare -a pred_files
declare -a dtypes

ind=0
method_list=('multiviewica' 'permica' 'groupica' 'control')

for method in ${method_list[@]}
do
    for dtype in "average" "subjectwise"
    do
        out_dirs[$ind]=$out_base$method"/100hz/time_window16-80/"
        real_files[$ind]=$real_base$method"/100hz/time_window16-80/shared_test.pkl"
        pred_files[$ind]=$pred_base$method"/100hz/time_window16-80/Y_test_predicted_"$dtype".pkl"
        dtypes[$ind]=$dtype
        ((ind=ind+1))
    done
done

out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
real_files=${real_files[$SLURM_ARRAY_TASK_ID]}
pred_files=${pred_files[$SLURM_ARRAY_TASK_ID]}
dtypes=${dtypes[$SLURM_ARRAY_TASK_ID]}

echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
echo out_dir: $out_dirs
echo pred_file: $pred_files
echo real_file: $real_files
echo dtype: $dtypes

cd  /home/akitaitsev/code/intersubject_generalization/linear/dataset1/generic_decoding/
python generic_decoding.py -real $real_files -pred $pred_files -regr_type $dtypes -out $out_dirs -sliding_window
