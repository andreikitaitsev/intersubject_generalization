#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=conv_enc_raw-lr
#SBATCH --mail-type=end
#SBATCH --mem=8000
#SBATCH --time=15:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# N JOBS = 4

batch_size=128
n_epochs=400
bpl=20
epta=10
out_dir='/scratch/akitaitsev/intersubject_generalization/dnn/conv_autoencoder/conv_autoencoder_raw-ver2/EEG/dataset1/50hz/ch-grid/'

enc_ch1=(16 32 64 128 )
enc_ch2=(32 64 128 256)
enc_ch3=(64 128 256 512)
enc_ch4=(128 256 512 1024)
enc_ch5=(256 512 1024 2048)

declare -a out_dirs

ind=0
for ind in $(seq 0 4)
do
    out_dirs[$ind]=$out_dir"/enc_ch1_"${enc_ch1[$ind]}"enc_ch2_"${enc_ch2[$ind]}"enc_ch3_"${enc_ch3[$ind]}"enc_ch4_"${enc_ch4[$ind]}"enc_ch5_"${enc_ch5[$ind]}"/"
done

out_dirs=${out_dirs[SLURM_ARRAY_TASK_ID]}

enc_ch1=${enc_ch1[SLURM_ARRAY_TASK_ID]}
enc_ch2=${enc_ch2[SLURM_ARRAY_TASK_ID]}
enc_ch3=${enc_ch3[SLURM_ARRAY_TASK_ID]}
enc_ch4=${enc_ch4[SLURM_ARRAY_TASK_ID]}
enc_ch5=${enc_ch5[SLURM_ARRAY_TASK_ID]}

echo enc_ch1: $enc_ch1
echo enc_ch2: $enc_ch2
echo enc_ch3: $enc_ch3
echo enc_ch4: $enc_ch4
echo enc_ch5: $enc_ch5
echo out_dir: $out_dirs

cd /home/akitaitsev/code/intersubject_generalization/dnn/conv_autoencoder/EEG/dataset1/
python conv_autoencoder_raw-ver2.py -gpu -out_dir $out_dirs -lr $lrs -batch_size $batch_size -n_epochs $n_epochs -scale 0 -normalize 0 
