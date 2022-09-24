#! /bin/bash
#SBATCH --job-name=gd_cv_controls
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=00:10:00
#SBATCH --qos=prio

# 4 jobs
out_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/generic_decoding/cv/"
real_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/featurematrices/100hz/time_window26-80/av_reps/"
pred_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/regression/cv/"

declare -a out_dirs
declare -a real_files
declare -a pred_files
dtype="subjectwise"

ind=0
method_list=('control10-cv0' 'control100-cv0' 'control10-cv1' 'control100-cv1' )
for method in ${method_list[@]}
do
    out_dirs[$ind]=$out_base$method"/100hz/time_window26-80/"
    real_files[$ind]=$real_base"/featurematrix_test.pkl"
    pred_files[$ind]=$pred_base$method"/100hz/time_window26-80/Y_test_predicted_"$dtype".pkl"
    ((ind=ind+1))
done

out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
real_files=${real_files[$SLURM_ARRAY_TASK_ID]}
pred_files=${pred_files[$SLURM_ARRAY_TASK_ID]}

echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
echo out_dir: $out_dirs
echo pred_file: $pred_files
echo real_file: $real_files

cd  /home/akitaitsev/code/intersubject_generalization/linear/dataset2/generic_decoding/
# indicate that the real data is raw (untransformed by IGA) EEG
python generic_decoding.py -real $real_files -pred $pred_files -out $out_dirs -cv 
