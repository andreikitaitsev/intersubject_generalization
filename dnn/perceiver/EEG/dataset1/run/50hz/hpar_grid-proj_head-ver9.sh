#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=hpar-9
#SBATCH --mail-type=end
#SBATCH --mem=8000
#SBATCH --time=70:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# N JOBS 4
srate=50
lr=0.0001
batch_size=128
n_epochs=1000
bpl=10
epta=1
clip_grad_norm=2
eeg_dir="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/dataset_matrices/50hz/time_window13-40/"
out_dir="/scratch/akitaitsev/intersubject_generalization/dnn/perceiver/EEG/dataset1/leftthomas/projection_head/"$srate"hz/hpar-grid-ver9/"
names=("simple-no_share_weights" "simple_share_weights" "complex_no_share_weights" "complex_share_weights")


# 4 configs: 0-simple, 1 - complex;  each with and without sharing of weights
lat_ar_dim=(200 200 200 200)
num_latent_dim=(50 50 100 100)
latent_head=(1 1 2 2)
depth=(1 1 1 1)
out_dim_ENC=(200 200 200 200)
out_dim_PH=(100 100 100 100)
cross_head=(1 1 1 1)
latent_dim_head=(64 64 256 256)
self_per_cross_attn=(2 2 2 2)
num_freq_band=(12 12 12 12)
share_weights=(1 0 1 0)


declare -a out_dirs
for ind in $(seq 0 3)
do
    out_dirs[$ind]=$out_dir"/"${names[$ind]}"/"
done

lat_ar_dim=${lat_ar_dim[$SLURM_ARRAY_TASK_ID]}
num_latent_dim=${num_latent_dim[$SLURM_ARRAY_TASK_ID]}
latent_head=${latent_head[$SLURM_ARRAY_TASK_ID]}
depth=${depth[$SLURM_ARRAY_TASK_ID]}
out_dim_ENC=${out_din_ENC[$SLURM_ARRAY_TASK_ID]}
out_dim_PH=${out_dim_PH[$SLURM_ARRAY_TASK_ID]}
cross_head=${cross_head[$SLURM_ARRAY_TASK_ID]}
cross_dim_head=${cross_dim_head[$SLURM_ARRAY_TASK_ID]}
latent_dim_head=${latent_dim_head[$SLURM_ARRAY_TASK_ID]}
self_per_cross_attn=${self_per_cross_attn[$SLURM_ARRAY_TASK_ID]}
num_freq_band=${num_freq_band[$SLURM_ARRAY_TASK_ID]}
share_weights=${share_weights[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}


echo lat_ar_dim: $lat_ar_dim
echo num_latent_dim: $num_latent_dim
echo latent_head: $latent_head
echo depth: $depth
echo out_dim_ENC: $out_dim_ENC
echo out_dimPH: $out_dim_PH
echo cross_head: $cross_head
echo cross_dim_head: $cross_dim_head
echo latent_dim_head: $latent_dim_head
echo self_per_cross_attn: $self_per_cross_attn
echo num_freq_band: $num_freq_band
echo share_weights: $share_weights
echo out_dir: $out_dirs

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/akitaitsev/anaconda3/lib/
cd /home/akitaitsev/code/intersubject_generalization/dnn/perceiver/EEG/dataset1/

python perceiver-projection_head-eeg-leftthomas.py -gpu -batch_size $batch_size -out_dir $out_dirs -lr $lr -bpl $bpl -epta $epta -n_epochs $n_epochs -perc_latent_array_dim $lat_ar_dim -perc_num_latent_dim $num_latent_dim -perc_latent_heads $latent_head -perc_depth $depth -out_dim_ENC $out_dim_ENC -out_dim_PH $out_dim_PH -perc_cross_heads $cross_head -perc_cross_dim_head $cross_dim_head -perc_latent_dim_head $latent_dim_head -perc_self_per_cross_attn $self_per_cross_attn -perc_num_freq_bands $num_freq_band -perc_share_weights $share_weights -clip_grad_norm $clip_grad_norm -pick_best_net_state -eeg_dir $eeg_dir
