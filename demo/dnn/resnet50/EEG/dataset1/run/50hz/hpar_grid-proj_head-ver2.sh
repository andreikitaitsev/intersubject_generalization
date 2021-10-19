#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=resnet_hpar-v2
#SBATCH --mail-type=end
#SBATCH --mem=8000
#SBATCH --time=70:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# gridsearch over perceiver with proj head params with lr=1e-4
# N JOBS = 5

# 50 hz

srate=50
# Non variable params
lr=0.0001
batch_size=128
n_epochs=300
bpl=10
epta=5
out_dir="/scratch/akitaitsev/intersubject_generalization/dnn/resnet50/EEG/dataset1/"$srate"hz/hpar-grid-ver2/"
names=("feature_dim")

# params to make gridsearch over
feature_dim=(64 128 200 512 2048)

declare -a out_dirs
declare -a feature_dims


ind=0
for par1 in ${feature_dim[@]}
do
    out_dirs[$ind]=$out_dir"/"${names[0]}$par1
    feature_dims[$ind]=$par5
    ((ind=ind+1))
done

out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
feature_dims=${feature_dims[$SLURM_ARRAY_TASK_ID]}

echo out_dirs: $out_dirs
echo feature_dims: $feature_dims

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/akitaitsev/anaconda3/lib/
cd /home/akitaitsev/code/intersubject_generalization/dnn/resnet50/EEG/dataset1/

python resnet50-eeg-leftthomas.py -gpu -batch_size $batch_size -out_dir $out_dirs -lr $lr -bpl $bpl -epta $epta -n_epochs $n_epochs -feature_dim $feature_dims -pick_best_net_state
