#! /bin/bash
#SBATCH --job-name=CIFAR-10
#SBATCH --mail-type=end
#SBATCH --mem=8000
#SBATCH --time=24:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


out_dir="/scratch/akitaitsev/intersubject_generalizeation/dnn/perceiver/trial/contrastive_loss-leftthomas-no_softmax/"
n_workers=0
batch_size=16
n_epochs=20
lr=0.01
cd /home/akitaitsev/code/intersubject_generalization/dnn/perceiver/trial/CIFAR/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/akitaitsev/anaconda3/lib/
python perceiver-cifar-contrastive_loss-leftthomas-no_softmax.py -out_dir $out_dir -n_workers $n_workers -batch_size $batch_size -learning_rate $lr -gpu -n_epochs $n_epochs -batch_per_loss 20 >./report.txt 2>&1