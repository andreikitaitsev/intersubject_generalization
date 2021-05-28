#! /bin/bash
#SBATCH --job-name=eeg_dnn
#SBATCH --mail-type=end
#SBATCH --mem=8000
#SBATCH --time=40:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# N Jobs = 4
declare -a lrates
declare -a out_dirs

batch_size=16
n_epochs=100
bpl=10
out_dir="/scratch/akitaitsev/intersubject_generalizeation/dnn/perceiver/trial2/EEG/zablo/"

ind=0
for lr in "0.05" "0.01" "0.005" "0.001"
do
    lrates[$ind]=$lr
    out_dirs[$ind]=$out_dir"lr"$lr"/"
    ((ind=ind+1))
done

lrates=${lrates[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}

echo lrates: $lrates
echo out_dir $out_dirs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/akitaitsev/anaconda3/lib/
cd /home/akitaitsev/code/intersubject_generalization/dnn/perceiver/trial2/EEG/
python perceiver-zablo.py -out_dir $out_dirs -batch_size $batch_size -gpu -batch_per_loss $bpl -n_epochs $n_epochs -lr $lrates
