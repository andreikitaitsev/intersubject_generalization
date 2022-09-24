#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=conv_enc_raw-lr
#SBATCH --mail-type=end
#SBATCH --mem=8000
#SBATCH --time=30:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# N JOBS = 7

lr=0.01
bpl=20
epta=10
out_dir='/scratch/akitaitsev/intersubject_generalization/dnn/conv_autoencoder/conv_autoencoder_raw-ver2/EEG/dataset1/50hz/batch_size-grid/'

batch_size_ar=(16 32 64 128 256 512 1024)

# set number if epochs so that the number of optimization iterations is the same
# for different batch sizes == 100 000 iterations

n_epochs=(200 400 850 1700 3400 7500 14000)

declare -a out_dirs
declare -a batch_sizes

ind=0
for batch_size in ${batch_size_ar[@]}
do
    batch_sizes[$ind]=$batch_size
    out_dirs[$ind]=$out_dir"/batch_size"$batch_size"/"
    ((ind=ind+1))
done

batch_sizes=${batch_sizes[SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[SLURM_ARRAY_TASK_ID]}
n_epochs=${n_epochs[SLURM_ARRAY_TASK_ID]}

echo batch_size: $bacth_sizes
echo out_dir: $out_dirs

cd /home/akitaitsev/code/intersubject_generalization/dnn/conv_autoencoder/EEG/dataset1/
python conv_autoencoder_raw-ver2.py -gpu -out_dir $out_dirs -lr $lr -batch_size $batch_sizes -scale 0 -normalize 0 -n_epochs $n_epochs
