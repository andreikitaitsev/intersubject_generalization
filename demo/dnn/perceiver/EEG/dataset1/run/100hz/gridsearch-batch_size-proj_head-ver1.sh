#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=grid_b-size100hz-v1
#SBATCH --mail-type=end
#SBATCH --mem=8000
#SBATCH --time=70:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# gridsearch over batch sizes with the best most shallow configuration of perceiver with proj 
# head params with lr=1e-4.
# number of optimization steps is eualized between epochs
# N JOBS = 4

# Non variable params
lr=0.0001
bpl=10
epta=5
out_dir="/scratch/akitaitsev/intersubject_generalization/dnn/perceiver/EEG/leftthomas/projection_head/100hz/gridsearch-batch_size-ver1/"
eeg_dir="/scratch/akitaitsev/intersubject_generalization/linear/dataset1/dataset_matrices/100hz/time_window26-80"

# create number of epochs so that number of optimization steps is
# the same between different batch sizes
n_iter=60000 # number of optimization iterations == n_epochs for contant batch size
dataset_size=7500 # number of train images per dataset
batch_size=(60 150 250 500)
n_opt_per_dataset=(150 50 30 15) # dataset_size / batch_size
n_epochs=(400 1200 2000 4000) # number of epochs for each batch size to have the same number of 
# optimization steps

perc_latent_array_dim=(200)
perc_num_latent_dim=(50)
perc_latent_head=(1)
perc_depth=(1)
out_dim_ENC=(200) # as if mvica with PCA 200
out_dim_PH=(100) # as if mvica with PCA 100

declare -a batch_sizes
declare -a out_dirs


ind=0
for par1 in ${batch_size[@]}
do
    batch_sizes[$ind]=$par1
    out_dirs[$ind]=$out_dir"/batch_size"$par1"/"
    ((ind=ind+1))
done

batch_sizes=${batch_sizes[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
n_epochs=${n_epochs[$SLURM_ARRAY_TASK_ID]}

echo batch_size: $batch_sizes
echo n_epochs: $n_epochs
echo out_dir: $out_dirs

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/akitaitsev/anaconda3/lib/
cd /home/akitaitsev/code/intersubject_generalization/dnn/perceiver/EEG/

python perceiver-projection_head-eeg-leftthomas.py -gpu -batch_size $batch_sizes -out_dir $out_dirs -lr $lr -bpl $bpl -epta $epta -n_epochs $n_epochs -perc_latent_array_dim $perc_latent_array_dim -perc_num_latent_dim $perc_num_latent_dim -perc_latent_heads $perc_latent_head -perc_depth $perc_depth -out_dim_ENC $out_dim_ENC -out_dim_PH $out_dim_PH -eeg_dir $eeg_dir

