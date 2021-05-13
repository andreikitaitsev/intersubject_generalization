#! /bin/bash
#SBATCH --job-name=eeg_dnn
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=24:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

# max batch size that fits into GPU memory
batch_size=32
out_dir="/scratch/akitaitsev/intersubject_generalizeation/dnn/perceiver/trial/EEG/2_gpu/"

cd /home/akitaitsev/code/intersubject_generalization/dnn/perceiver/trial/EEG/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/akitaitsev/anaconda3/lib/
python perceiver.py -out_dir $out_dir -batch_size $batch_size -gpu
