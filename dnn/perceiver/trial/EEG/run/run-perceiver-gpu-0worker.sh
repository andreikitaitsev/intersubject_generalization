#! /bin/bash
#SBATCH --job-name=eeg_dnn
#SBATCH --mail-type=end
#SBATCH --mem=6000
#SBATCH --time=24:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

n_workers=0
# max batch size that fits into GPU memory
batch_size=16
out_dir="/scratch/akitaitsev/intersubject_generalizeation/dnn/perceiver/trial/EEG/1_gpu_/"

cd /home/akitaitsev/code/intersubject_generalization/dnn/perceiver/trial/EEG/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/akitaitsev/anaconda3/lib/
python perceiver.py -out_dir $out_dir -n_workers $n_workers -batch_size $batch_size -n_gpu 1
