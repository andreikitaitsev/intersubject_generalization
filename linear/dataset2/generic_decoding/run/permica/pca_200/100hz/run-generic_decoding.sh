#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=generic_decoding_time_window0-40
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=00:10:00
#SBATCH --qos=prio
# N JONS = 4

# 100 hz veriosn

srate=100
out_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/generic_decoding/permica/pca_200/"$srate"hz/time_window"
real_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/intersubject_generalization/permica/pca_200/"$srate"hz/time_window"
pred_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/regression/permica/pca_200/"$srate"hz/time_window"

declare -a out_dirs
declare -a real_files
declare -a pred_files
declare -a dtypes

ind=0
for dtype in "subjectwise" "average"
do
    fot t in "0-80" "26-80"
    do
        out_dris[$ind]=$out_base$t"/av_reps/"
        real_files[$ind]=$real_base$t"/av_reps/shared_test.pkl"
        pred_files[$ind]=$pred_base$T"av_reps/shared_test_predicted_"$dtype".pkl"
        dtypes[$ind]=$dtype
        ((ind=ind+1))
   done
done

out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
real_files=${real_files[$SLURM_ARRAY_TASK_ID]}
pred_files=${pred_files[$SLURM_ARRAY_TASK_ID]}
dtypes=${dtypes[$SLURM_ARRAY_TASK_ID]}

echo output_dir: $out_dirs
echo dtype: $dtypes

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset2/generic_decoding/
echo Running generic decoding on control data with pca=200
python generic_decoding.py -real $real_files -pred $pred_files -d_type $dtypes -out $out_dirs
