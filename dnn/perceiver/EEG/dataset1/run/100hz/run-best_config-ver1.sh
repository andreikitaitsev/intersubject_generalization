#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=best_conf_100hz
#SBATCH --mail-type=end
#SBATCH --mem=2000
#SBATCH --time=70:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# run perceiver with best config found on dataset1
# N JOBS = 1

# 100 hz

srate=100
tw="26-80"

# Non variable params
lr=0.0001
batch_size=128
n_epochs=1000
bpl=10
epta=5
out_dir="/scratch/akitaitsev/intersubject_generalization/dnn/perceiver/EEG/dataset1/leftthomas/projection_head/"$srate"hz/best_config-ver1/"
eeg_dir="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/dataset_matrices/"$srate"hz/time_window"$tw"/"
perc_latent_array_dim=200
perc_num_latent_dim=50
perc_latent_head=2
perc_depth=1
out_dim_ENC=200 # as if mvica with PCA 200
out_dim_PH=100 # as if mvica with PCA 100

echo perc_latent_array_dims: $perc_latent_array_dim
echo perc_num_latent_dims: $perc_num_latent_dim
echo perc_latent_heads: $perc_latent_head
echo perc_depths: $perc_depth
echo out_dim_ENCs $out_dim_ENC
echo out_dim_PHs: $out_dim_PH
echo out_dir: $out_dir

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/akitaitsev/anaconda3/lib/
cd /home/akitaitsev/code/intersubject_generalization/dnn/perceiver/EEG/dataset2/

python perceiver-projection_head-eeg-leftthomas.py -gpu -batch_size $batch_size -out_dir $out_dir -lr $lr -bpl $bpl -epta $epta -n_epochs $n_epochs -perc_latent_array_dim $perc_latent_array_dim -perc_num_latent_dim $perc_num_latent_dim -perc_latent_heads $perc_latent_head -perc_depth $perc_depth -out_dim_ENC $out_dim_ENC -out_dim_PH $out_dim_PH -eeg_dir $eeg_dir -pick_best_net_state

