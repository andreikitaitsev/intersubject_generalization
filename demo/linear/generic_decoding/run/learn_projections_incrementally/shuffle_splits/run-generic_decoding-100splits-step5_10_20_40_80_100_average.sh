#! /bin/bash
#SBATCH --job-name=gen_dec
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=00:10:00
#SBATCH --qos=prio

# N JOBS = 600

step_list=(5 10 20 40 80 100)
nsplits=100
nshuffles=100
method="multiviewica"
prepr="pca"
n_comps=200
regr_type="average"

out_base="/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/learn_projections_incrementally/"\
"shuffle_splits/"$nsplits"splits/"$nshuffles"shuffles/"$method"/"$prepr"/"$n_comps"/"
pred_base="/scratch/akitaitsev/intersubject_generalizeation/linear/regression/learn_projections_incrementally/shuffle_splits/"\
$nsplits"splits/"$nshuffles"shuffles/"$method"/"$prepr"/"$n_comps"/"
real_base="/scratch/akitaitsev/intersubject_generalizeation/linear/intersubject_generalization/learn_projections_incrementally/"\
"shuffle_splits/"$nsplits"splits/"$nshuffles"shuffles/"$method"/"$prepr"/"$n_comps"/"
rname="shared_test.pkl"
prname_base="Y_test_predicted_"

declare -a steps
declare -a real_files
declare -a pred_files
declare -a out_dirs
declare -a shuffles

ind=0
for shuffle in $(seq 0 $((nshuffles-1)))
do
    for step in ${step_list[*]}
    do
        real_files[$ind]=$real_base"/50hz/time_window13-40/shuffle_"$shuffle"/step_"$(printf '%d' $step)"/"$rname
        pred_files[$ind]=$pred_base"/50hz/time_window13-40/shuffle_"$shuffle"/step_"$(printf '%d' $step)"/"$prname_base$regr_type".pkl"
        out_dirs[$ind]=$out_base"/50hz/time_window13-40/shuffle_"$shuffle"/step_"$(printf '%d' $step)"/"
        steps[$ind]=$step
        shuffles[$ind]=$shuffle
        ((ind=ind+1))
    done
done

real_files=${real_files[$SLURM_ARRAY_TASK_ID]}
pred_files=${pred_files[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
steps=${steps[$SLURM_ARRAY_TASK_ID]}
shuffles=${shuffles[$SLURM_ARRAY_TASK_ID]}

echo SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID
echo real_file: $real_files
echo pred_file: $pred_files
echo out_dir: $out_dirs
echo method: $method
echo prepr: $prepr
echo n_comp: $n_comps
echo shuffle: $shuffles
echo step: $steps
echo regr_type: $regr_type

cd /home/akitaitsev/code/intersubject_generalization/linear/generic_decoding
python generic_decoding.py -real $real_files -pred $pred_files -out $out_dirs -regr_type $regr_type -learn_pr_incr