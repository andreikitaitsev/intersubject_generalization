#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=generic_decoding_time_window13-40
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=00:10:00
#SBATCH --qos=prio

# 4 tasks
dir_base="/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/raw/time_window"

declare -a out_dirs
declare -a res
declare -a cor

ind=0
for time_window in "0-40" "13-40"
do
    for type_ in "average" "subjectwise"
    do
        out_dirs[$ind]=$dir_base$time_window
        res[$ind]=$dir_base$time_window"/generic_decoding_results_"$type_".pkl"
        cor[$ind]=$dir_base$time_window"/generic_decoding_correlations_"$type_".pkl"
        ((ind=ind+1))
    done
done

out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
res=${res[$SLURM_ARRAY_TASK_ID]}
cor=${cor[$SLURM_ARRAY_TASK_ID]}

cd /home/akitaitsev/code/intersubject_generalization/linear/generic_decoding/
echo SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID
echo results filepath: $res

python plots_generic_decoding.py -res $res -cor $cor -out_dir $out_dirs
