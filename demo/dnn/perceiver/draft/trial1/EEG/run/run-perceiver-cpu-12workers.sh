#! /bin/bash
#SBATCH --job-name=CIFAR-10
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=48:00:00
#SBATCH --qos=prio
#SBATCH --cpus-per-task=12

n_workers=12
batch_size=64
out_dir="/scratch/akitaitsev/intersubject_generalizeation/dnn/perceiver/trial/EEG/params1/"

cd /home/akitaitsev/code/intersubject_generalization/dnn/perceiver/trial/EEG/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/akitaitsev/anaconda3/lib/
python perceiver.py -out_dir $out_dir -n_workers $n_workers -batch_size $batch_size 
