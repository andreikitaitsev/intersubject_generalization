#! /bin/bash
#SBATCH --job-name=CIFAR-10
#SBATCH --mail-type=end
#SBATCH --mem=8000
#SBATCH --time=24:00:00
#SBATCH --qos=prio
#SBATCH --cpus-per-task=12

n_workers=12
batch_size=4
out_dir="/scratch/akitaitsev/intersubject_generalizeation/dnn/perceiver/trial/"$n_workers"cpu/"

cd /home/akitaitsev/code/intersubject_generalization/dnn/perceiver/trial
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/akitaitsev/anaconda3/lib/
python test-perceiver-cifar.py -out_dir $out_dir -n_workers $n_workers -batch_size $batch_size
