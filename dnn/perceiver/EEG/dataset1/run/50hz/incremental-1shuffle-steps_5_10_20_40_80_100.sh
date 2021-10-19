#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=incr1shuffle
#SBATCH --mail-type=end
#SBATCH --mem=8000
#SBATCH --time=70:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# N JOBS 6

srate=50
lr=0.0001
batch_size=128
n_epochs=1500
bpl=10
epta=5
clip_grad_norm=2
nsplits=100
step_list=(5 10 20 40 80 100) # percents of training data used
eeg_dir="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/dataset_matrices/50hz/time_window13-40/"
out_dir="/scratch/akitaitsev/intersubject_generalization/dnn/perceiver/EEG/dataset1/leftthomas/projection_head/"$srate"hz/incr-1shuffle-100splits/"


lat_ar_dim=200
num_latent_dim=50
latent_head=1
depth=1
out_dim_ENC=200
out_dim_PH=100
cross_head=1
cross_dim_head=64
latent_dim_head=64
self_per_cross_attn=2
num_freq_band=12
share_weights=0


declare -a out_dirs
declare -a seeds
declare -a steps

ind=0
seed=0
for step in ${step_list[@]} #start step stop
do
    seeds[$ind]=$seed
    steps[$ind]=$step
    out_dirs[$ind]=$out_dir"/step"$step"/seed"$seed"/"
    ((ind=ind+1))
    ((seed=seed+1))
done


seeds=${seeds[$SLURM_ARRAY_TASK_ID]}
steps=${steps[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}


echo step: $steps
echo seed: $seeds
echo out_dir: $out_dirs

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/akitaitsev/anaconda3/lib/
cd /home/akitaitsev/code/intersubject_generalization/dnn/perceiver/EEG/dataset1/

python perceiver-projection_head-eeg-leftthomas-incremental.py -gpu -batch_size $batch_size -out_dir $out_dirs -lr $lr -bpl $bpl -epta $epta -n_epochs $n_epochs -perc_latent_array_dim $lat_ar_dim -perc_num_latent_dim $num_latent_dim -perc_latent_heads $latent_head -perc_depth $depth -out_dim_ENC $out_dim_ENC -out_dim_PH $out_dim_PH -perc_cross_heads $cross_head -perc_cross_dim_head $cross_dim_head -perc_latent_dim_head $latent_dim_head -perc_self_per_cross_attn $self_per_cross_attn -perc_num_freq_bands $num_freq_band -perc_share_weights $share_weights -clip_grad_norm $clip_grad_norm -pick_best_net_state -eeg_dir $eeg_dir -seed $seeds -step $steps -nsplits $nsplits
