#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=cv_par
#SBATCH --mail-type=end
#SBATCH --mem=40000
#SBATCH --time=24:00:00
#SBATCH --qos=prio

# Run intersubject generalization in leave -one-out cross validation
# on different methods with pca as preprocessor

# N JOBS = 63

n_splits=7 # leave one subject out
inp_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/featurematrices/50hz/time_window13-40/"
out_dir_base="/scratch/akitaitsev/intersubject_generalizeation/linear/intersubject_generalization/cross_val/"

declare -a methods 
declare -a n_comps
declare -a out_dirs
declare -a splits2use
prepr="pca"

ind=0
for method in "multiviewica" "permica" "groupica"
do
    for n_comp in 50 200 400
    do
        for split2use in 0 1 2 3 4 5 6
        do
            splits2use[$ind]=$split2use
            methods[$ind]=$method
            n_comps[$ind]=$n_comp
            out_dirs[$ind]=$out_dir_base$method"/main/"$prepr"/"$n_comp"/50hz/time_window13-40/"
            ((ind=ind+1))
        done
    done
done

out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}            
methods=${methods[$SLURM_ARRAY_TASK_ID]}
n_comps=${n_comps[$SLURM_ARRAY_TASK_ID]}
splits2use=${splits2use[$SLURM_ARRAY_TASK_ID]}

echo SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID
echo method: $methods
echo n_comp: $n_comps
echo split2use: $splits2use
echo output_dir: $out_dirs

cd /home/akitaitsev/code/intersubject_generalization/linear/intersubject_generalization
python cross_val_single_split.py -inp $inp_dir -out $out_dirs -method $methods -dim_reduction $prepr -n_comp $n_comps -n_splits $n_splits \
    -split2use $splits2use
