#! /bin/bash
#SBATCH --job-name=gen_dec
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=00:10:00
#SBATCH --qos=prio

# N JOBS = 300

step_list=(10 20 30 40 50 60 70 80 90 100)
nsplits=100
nshuffles=10
regr_type="average"

method_list=("multiviewica" "groupica" "permica")
out_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/generic_decoding/saturation_profile-different_methods/"
pred_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/regression/saturation_profile-different_methods/"
real_base="/scratch/akitaitsev/intersubject_generalization/linear/dataset2/intersubject_generalization/saturation_profile-different_methods/"

rname="shared_test.pkl"
prname_base="Y_test_predicted_"

declare -a steps
declare -a real_files
declare -a pred_files
declare -a out_dirs
declare -a shuffles
declare -a methods

ind=0
for method in ${method_list[@]}
do
    for shuffle in $(seq 0 $((nshuffles-1)))
    do
        for step in ${step_list[*]}
        do
            methods[$ind]=$method
            real_files[$ind]=$real_base"/"$method"/50hz/shuffle_"$shuffle"/step_"$(printf '%d' $step)"/"$rname
            pred_files[$ind]=$pred_base"/"$method"/50hz/shuffle_"$shuffle"/step_"$(printf '%d' $step)"/"$prname_base$regr_type".pkl"
            out_dirs[$ind]=$out_base"/"$method"/50hz/shuffle_"$shuffle"/step_"$(printf '%d' $step)"/"
            steps[$ind]=$step
            shuffles[$ind]=$shuffle
            ((ind=ind+1))
        done
    done
done

real_files=${real_files[$SLURM_ARRAY_TASK_ID]}
pred_files=${pred_files[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
steps=${steps[$SLURM_ARRAY_TASK_ID]}
shuffles=${shuffles[$SLURM_ARRAY_TASK_ID]}
methods=${methods[$SLURM_ARRAY_TASK_ID]}

echo SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID
echo method: $methods
echo real_file: $real_files
echo pred_file: $pred_files
echo out_dir: $out_dirs
echo shuffle: $shuffles
echo step: $steps
echo regr_type: $regr_type

cd /home/akitaitsev/code/intersubject_generalization/linear/dataset1/generic_decoding
python generic_decoding.py -real $real_files -pred $pred_files -out $out_dirs -regr_type $regr_type -learn_pr_incr
