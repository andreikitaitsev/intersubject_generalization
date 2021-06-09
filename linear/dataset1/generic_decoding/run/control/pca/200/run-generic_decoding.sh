#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=gen_dec
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=00:10:00
#SBATCH --qos=prio


### 4 Jobs
out_base="/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/control/pca/200/time_window"
real_base="/scratch/akitaitsev/intersubject_generalizeation/linear/intersubject_generalization/control/pca/200/time_window"
pred_base="/scratch/akitaitsev/intersubject_generalizeation/linear/regression/control/pca/200/time_window"

declare -a out_dirs
declare -a real_files
declare -a pred_files 
declare -a dtypes 
ind=0
for dtype in "subjectwise" "average"
do
    for t in "0-40" "13-40" 
    do
        out_dirs[$ind]=$out_base$t
        real_files[$ind]=$real_base$t"/shared_test.pkl"
        pred_files[$ind]=$pred_base$t"/Y_test_predicted_"$dtype".pkl"
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

cd /home/akitaitsev/code/intersubject_generalization/linear/generic_decoding/
echo Running generic decoding on control data with pca=200
python generic_decoding.py -real $real_files -pred $pred_files -regr_type $dtypes -out $out_dirs
