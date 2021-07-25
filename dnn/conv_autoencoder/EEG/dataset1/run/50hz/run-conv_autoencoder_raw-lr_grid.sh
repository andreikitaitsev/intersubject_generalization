#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=conv_enc_raw-lr
#SBATCH --mail-type=end
#SBATCH --mem=8000
#SBATCH --time=15:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# N JOBS = 6

batch_size=128
n_epochs=400
bpl=20
epta=10
out_dir='/scratch/akitaitsev/intersubject_generalization/dnn/conv_autoencoder/conv_autoencoder_raw/EEG/dataset1/50hz/lr-grid/'

lr_ar=(0.1 0.01 0.001 0.0001 0.00001 0.000001)

declare -a out_dirs
declare -a lrs

ind=0
for lr in ${lr_ar[@]}
do
    lrs[$ind]=$lr
    out_dirs[$ind]=$out_dir"/lr"$lr"/"
    ((ind=ind+1))
done

lrs=${lrs[SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[SLURM_ARRAY_TASK_ID]}

echo lr: $lrs
echo out_dir: $out_dirs

cd /home/akitaitsev/code/intersubject_generalization/dnn/conv_autoencoder/EEG/dataset1/
python conv_autoencoder_raw.py -gpu -out_dir $out_dirs -lr $lrs -batch_size $batch_size -n_epochs $n_epochs -scale 0 -normalize 0 
