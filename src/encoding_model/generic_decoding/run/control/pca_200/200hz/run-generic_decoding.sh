#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=gen_dec
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=00:20:00
#SBATCH --qos=prio

# 200 hz version

### 4 Jobs
srate=200
out_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/generic_decoding/control/pca_200/"$srate"hz/time_window"
real_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/intersubject_generalization/control/pca_200/"$srate"hz/time_window"
pred_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/regression/control/pca_200/"$srate"hz/time_window"

declare -a out_dirs
declare -a real_files
declare -a pred_files 
declare -a dtypes 
ind=0
for dtype in "subjectwise" "average"
do
    for t in "0-160" "52-160" 
    do
        out_dirs[$ind]=$out_base$t"/av_reps/"
        real_files[$ind]=$real_base$t"/av_reps/shared_test.pkl"
        pred_files[$ind]=$pred_base$t"/av_reps/Y_test_predicted_"$dtype".pkl"
        dtypes[$ind]=$dtype
        ((ind=ind+1))
    done
done


### Extracting the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID

out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
real_files=${real_files[$SLURM_ARRAY_TASK_ID]}
pred_files=${pred_files[$SLURM_ARRAY_TASK_ID]}
dtypes=${dtypes[$SLURM_ARRAY_TASK_ID]}

echo output_dir: $out_dirs
echo dtype: $dtypes

sleep 10

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset2/generic_decoding/
echo Running generic decoding on control data with pca=200
python generic_decoding.py -real $real_files -pred $pred_files -regr_type $dtypes -out $out_dirs