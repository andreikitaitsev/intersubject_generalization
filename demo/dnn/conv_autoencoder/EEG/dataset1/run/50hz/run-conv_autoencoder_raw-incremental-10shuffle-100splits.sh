#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=ch_grid-conv_enc_raw
#SBATCH --mail-type=end
#SBATCH --mem=8000
#SBATCH --time=15:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# N JOBS = 60

srate=50
batch_size=16
n_epochs=100
bpl=20
epta=1
out_dir='/scratch/akitaitsev/intersubject_generalization/dnn/conv_autoencoder/conv_autoencoder_raw/EEG/dataset1/'$srate'hz/incr-10shuffle-100splits/'
eeg_dir='/scratch/akitaitsev/intersubject_generalization/linear/dataset1/dataset_matrices/50hz/time_window13-40/'
lr=0.01
nsplits=100
step_list=(5 10 20 40 80 100)

enc_ch1=128
enc_ch2=256
enc_ch3=512
dec_ch1=512
dec_ch2=256

declare -a out_dirs
declare -a seeds
declare -a steps

ind=0
for step in ${step_list[@]}
do
    for seed in $(seq 0 9)
    do
    out_dirs[$ind]=$out_dir"/step"$step"/seed"$seed"/"
    seeds[$ind]=$seed
    steps[$ind]=$step
    ((ind=ind+1))
    done
done

seeds=${seeds[$SLURM_ARRAY_TASK_ID]}
steps=${steps[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}

echo step: $steps
echo seed: $seeds
echo out_dir: $out_dirs

cd /home/akitaitsev/code/intersubject_generalization/dnn/conv_autoencoder/EEG/dataset1/

python conv_autoencoder_raw-incremental.py -gpu -out_dir $out_dirs -lr $lr -batch_size $batch_size -n_epochs $n_epochs -scale 0 -normalize 0 -enc_chs $enc_ch1 $enc_ch2 $enc_ch3 -dec_chs $dec_ch1 $dec_ch2 -eeg_dir $eeg_dir -nsplits $nsplits -step $steps -seed $seeds
