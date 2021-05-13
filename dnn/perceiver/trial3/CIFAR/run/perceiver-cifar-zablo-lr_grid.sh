#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=lr_grid_zablo
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=40:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# N Jobs = 4
declare -a lrates
declare -a out_dirs

batch_size=64
n_epochs=100
bpl=50
out_dir="/scratch/akitaitsev/intersubject_generalization/dnn/perceiver/trial3/CIFAR/zablo/"

ind=0
lr_ar=(0.001 0.0001 0.00001 0.000001)
for lr in ${lr_ar[@]}  # 1e-3 1e-4 1e-5 1e-6
do
    lrates[$ind]=$lr
    out_dirs[$ind]=$out_dir"lr"$lr"/"
    ((ind=ind+1))
done

lrates=${lrates[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}

echo lrates: $lrates
echo out_dir: $out_dirs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/akitaitsev/anaconda3/lib/
cd /home/akitaitsev/code/intersubject_generalization/dnn/perceiver/trial3/CIFAR/
python perceiver-cifar-zablo.py -out_dir $out_dirs -batch_size $batch_size -gpu -bpl $bpl -n_epochs $n_epochs -lr $lrates
