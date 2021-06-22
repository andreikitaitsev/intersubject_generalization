#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=share_wieghts_hpar_grid-5
#SBATCH --mail-type=end
#SBATCH --mem=8000
#SBATCH --time=70:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# gridsearch over perceiver with proj head params with lr=1e-4
# check for weight_tie_layers

# N JOBS = 1

declare -a out_dirs
# Non variable params
lr=0.0001
batch_size=128
n_epochs=1000
bpl=10
epta=5
out_dir="/scratch/akitaitsev/intersubject_generalization/dnn/perceiver/EEG/dataset1/leftthomas/projection_head/50hz/hpar-grid-ver5/"
names=("perc_latent_array_dim" "perc_num_latent_dim" "perc_latent_heads" "perc_depth" "out_dim_ENC" "out_dim_PH")

# params to make gridsearch over
perc_latent_array_dim=200
perc_num_latent_dim=50
perc_latent_head=2
perc_depth=1
out_dim_ENC=200 # as if mvica with PCA 200
out_dim_PH=100 # as if mvica with PCA 100



export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/akitaitsev/anaconda3/lib/
cd /home/akitaitsev/code/intersubject_generalization/dnn/perceiver/EEG/dataset1/

python perceiver-projection_head-eeg-leftthomas.py -gpu -batch_size $batch_size -out_dir $out_dir -lr $lr -bpl $bpl -epta $epta -n_epochs $n_epochs -perc_latent_array_dim $perc_latent_array_dim -perc_num_latent_dim $perc_num_latent_dim -perc_latent_heads $perc_latent_head -perc_depth $perc_depth -out_dim_ENC $out_dim_ENC -out_dim_PH $out_dim_PH -perc_share_weights

