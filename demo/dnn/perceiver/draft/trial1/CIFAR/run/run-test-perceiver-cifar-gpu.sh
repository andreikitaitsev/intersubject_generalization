#! /bin/bash
#SBATCH --job-name=CIFAR-10
#SBATCH --mail-type=end
#SBATCH --mem=8000
#SBATCH --time=24:00:00
#SBATCH --qos=prio
#SBATCH --gres=gpu:1


out_dir="/scratch/akitaitsev/intersubject_generalizeation/dnn/perceiver/trial/1gpu/"
n_workers=1
batch_size=4

cd /home/akitaitsev/code/intersubject_generalization/dnn/perceiver/trial
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/akitaitsev/anaconda3/lib/
python test-perceiver-cifar.py -out_dir $out_dir -n_workers $n_workers -batch_size $batch_size -gpu >./report.txt 2>&1
