#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=resnet_lr
#SBATCH --mail-type=end
#SBATCH --mem=8000
#SBATCH --time=30:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# gridsearch for resnet 18 with projection head with the default config
# N JOBS = 5

srate=50
batch_size=128
n_epochs=600
bpl=10
epta=5
out_dir="/scratch/akitaitsev/intersubject_generalization/dnn/resnet18/EEG/dataset1/"$srate"hz/gridsearch-lr/"
lr_array=(0.01 0.001 0.0001 0.00001 0.000001)

declare -a out_dirs
declare -a lrs


ind=0
for lr in ${lr_array[@]}
do
   out_dirs[$ind]=$out_dir"lr-"$lr
   lrs[$ind]=$lr
   ((ind=ind+1))
done

lrs=${lrs[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}

echo out_dir: $out_dirs
echo lrs: $lrs

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/akitaitsev/anaconda3/lib/
cd /home/akitaitsev/code/intersubject_generalization/dnn/resnet18/EEG/

python resnet18-eeg.py -gpu -batch_size $batch_size -out_dir $out_dirs -lr $lrs -bpl $bpl -epta $epta -n_epochs $n_epochs 
