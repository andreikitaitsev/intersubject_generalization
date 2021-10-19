#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=ch_grid-conv_enc_raw
#SBATCH --mail-type=end
#SBATCH --mem=8000
#SBATCH --time=15:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# N JOBS = 6

batch_size=16
n_epochs=400
bpl=20
epta=10
out_dir='/scratch/akitaitsev/intersubject_generalization/dnn/conv_autoencoder/conv_autoencoder_raw/EEG/dataset1/50hz/ch-grid/'
lr=0.01

enc_ch1=(8 16 32 64 128 256)
enc_ch2=(16 32 64 128 256 512)
enc_ch3=(32 64 128 256 512 1024)
dec_ch1=(32 64 128 256 512 1024)
dec_ch2=(16 32 64 128 256 512)

declare -a out_dirs

ind=0
for ind in $(seq 0 4)
do
    out_dirs[$ind]=$out_dir"/enc_ch1_"${enc_ch1[$ind]}"enc_ch2_"${enc_ch2[$ind]}"enc_ch3_"${enc_ch3[$ind]}"dec_ch1_"${dec_ch1[$ind]}"dec_ch2_"${dec_ch2[$ind]}"/"
done

out_dirs=${out_dirs[SLURM_ARRAY_TASK_ID]}
enc_ch1=${enc_ch1[SLURM_ARRAY_TASK_ID]}
enc_ch2=${enc_ch2[SLURM_ARRAY_TASK_ID]}
enc_ch3=${enc_ch3[SLURM_ARRAY_TASK_ID]}
dec_ch1=${dec_ch1[SLURM_ARRAY_TASK_ID]}
dec_ch2=${dec_ch2[SLURM_ARRAY_TASK_ID]}


echo out_dir: $out_dirs

cd /home/akitaitsev/code/intersubject_generalization/dnn/conv_autoencoder/EEG/dataset1/

python conv_autoencoder_raw.py -gpu -out_dir $out_dirs -lr $lr -batch_size $batch_size -n_epochs $n_epochs -scale 0 -normalize 0 -enc_chs $enc_ch1 $enc_ch2 $enc_ch3 -dec_chs $dec_ch1 $dec_ch2

