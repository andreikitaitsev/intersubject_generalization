#! /bin/bash
#SBATCH --job-name=gen_dec
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=00:10:00
#SBATCH --qos=prio

# 16 jobs
dir_base="/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/"
method="multiviewica"

declare -a out_dir
declare -a res
declare -a cor

ind=0
for prepr in "pca" "srm"
do
    for n_comp in 10 50 200 400
    do
        for dtype in "average" "subjectwise"
        do
            out_dir[$ind]=$dir_base$method"/main/"$prepr"/"$n_comp"/50hz/time_window13-40/"
            res[$ind]=$dir_base$method"/main/"$prepr"/"$n_comp"/50hz/time_window13-40/generic_decoding_results_"$dtype".pkl"
            cor[$ind]=$dir_base$method"/main/"$prepr"/"$n_comp"/50hz/time_window13-40/generic_decoding_correlations_"$dtype".pkl"
            ((ind=ind+1))
        done
    done
done

out_dir=${out_dir[$SLURM_ARRAY_TASK_ID]}
res=${res[$SLURM_ARRAY_TASK_ID]}
cor=${cor[$SLURM_ARRAY_TASK_ID]}

cd /home/akitaitsev/code/intersubject_generalization/linear/generic_decoding/
echo SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID
echo results filepath: $res

python plots_generic_decoding.py -res $res -cor $cor -out_dir $out_dir
