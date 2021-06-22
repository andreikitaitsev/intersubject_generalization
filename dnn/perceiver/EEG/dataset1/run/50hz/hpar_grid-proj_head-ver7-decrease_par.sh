#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=hpar_grid-v4
#SBATCH --mail-type=end
#SBATCH --mem=8000
#SBATCH --time=70:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# in hpar-grid-ver6.sh we compared default parameters and increased parameters:
# cross_head: 1(def), 3
# cross_dim_head: 64(def), 128
# latent_dim_head: 64 (def), 128
# self_per_cross_attn: 2 (def), 4
# num_freq_bands: 6(def), 12

# This script decreases the parameters compared to default based on results of hpar-ver6.sh
#(keep only those params with which acc continued to increase):
# cross_head: 1(def), 1
# cross_dim_head: 64(def), 32
# latent_dim_head: 64 (def), 32
# self_per_cross_attn: 2 (def), 1
# num_freq_bands: 6(def), 3

# N JOBS = 16

srate=50
lr=0.0001
batch_size=128
n_epochs=1200
bpl=10
epta=5
out_dir="/scratch/akitaitsev/intersubject_generalization/dnn/perceiver/EEG/dataset1/leftthomas/projection_head/"$srate"hz/hpar-grid-ver7-decrease_par/"
names=("cross_head" "cross_dim_head" "latent_dim_head" "self_per_cross_attn" "num_freq_band")

# estimated best params
perc_latent_array_dim=200
perc_num_latent_dim=50
perc_latent_head=2
perc_depth=1
out_dim_ENC=200 # as if mvica with PCA 200
out_dim_PH=100 # as if mvica with PCA 100

# params to make gridsearch over: enhance compared to default values
# (1st value -default, 2nd -enhanced)

cross_head=(1)
cross_dim_head=(32 64)
latent_dim_head=(32 64)
self_per_cross_attn=(1 2)
num_freq_band=(3 6)

declare -a out_dirs
declare -a cross_heads
declare -a cross_dim_heads
declare -a latent_dim_heads
declare -a self_per_cross_attns
declare -a num_freq_bands



ind=0
for par1 in ${cross_head[@]}
do
    for par2 in ${cross_dim_head[@]}
    do
        for par3 in ${latent_dim_head[@]}
        do
            for par4 in ${self_per_cross_attn[@]}
            do
                for par5 in ${num_freq_band[@]}
                do
                    out_dirs[$ind]=$out_dir"/"${names[0]}$par1"_"${names[1]}$par2"_"${names[2]}$par3"_"${names[3]}$par4"_"${names[4]}$par5
                    cross_heads[$ind]=$par1
                    cross_dim_heads[$ind]=$par2
                    latent_dim_heads[$ind]=$par3
                    self_per_cross_attns[$ind]=$par4
                    num_freq_bands[$ind]=$par5
                    ((ind=ind+1))
                done
            done
        done
    done
done

cross_heads=${cross_heads[$SLURM_ARRAY_TASK_ID]}
cross_dim_heads=${cross_dim_heads[$SLURM_ARRAY_TASK_ID]}
latent_dim_heads=${latent_dim_heads[$SLURM_ARRAY_TASK_ID]}
self_per_cross_attns=${self_per_cross_attns[$SLURM_ARRAY_TASK_ID]}
num_freq_bands=${num_freq_bands[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}

echo cross_heads: $cross_heads
echo cross_dim_heads: $cross_dim_heads
echo latent_dim_heads: $latent_dim_heads
echo self_per_cross_attns: $self_per_cross_attns
echo num_freq_bands: $num_freq_bands
echo out_dir: $out_dirs

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/akitaitsev/anaconda3/lib/
cd /home/akitaitsev/code/intersubject_generalization/dnn/perceiver/EEG/dataset1/

python perceiver-projection_head-eeg-leftthomas.py -gpu -batch_size $batch_size -out_dir $out_dirs -lr $lr -bpl $bpl -epta $epta -n_epochs $n_epochs -perc_latent_array_dim $perc_latent_array_dim -perc_num_latent_dim $perc_num_latent_dim -perc_latent_heads $perc_latent_head -perc_depth $perc_depth -out_dim_ENC $out_dim_ENC -out_dim_PH $out_dim_PH -perc_cross_heads $cross_heads -perc_cross_dim_head $cross_dim_heads -perc_latent_dim_head $latent_dim_heads -perc_self_per_cross_attn $self_per_cross_attns -perc_num_freq_bands $num_freq_bands
