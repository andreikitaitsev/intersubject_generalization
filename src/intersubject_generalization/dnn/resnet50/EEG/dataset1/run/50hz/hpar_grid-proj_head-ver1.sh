#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=resnet_hpar-v1
#SBATCH --mail-type=end
#SBATCH --mem=8000
#SBATCH --time=70:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# gridsearch over perceiver with proj head params with lr=1e-4
# N JOBS = 16

# 50 hz

srate=50
# Non variable params
lr=0.0001
batch_size=128
n_epochs=300
bpl=10
epta=5
out_dir="/scratch/akitaitsev/intersubject_generalization/dnn/resnet50/EEG/dataset1/"$srate"hz/hpar-grid-ver1/"
names=("out_ch" "kernel_size" "proj_head_inp_dim" "proj_head_intermediate_dim" "feature_dim")

# params to make gridsearch over
out_ch=(64)
kernel_size=(3 6)
proj_head_inp_dim=(128 2048)
proj_head_intermediate_dim=(128 2048)
feature_dim=(200 1024)

declare -a out_dirs
declare -a out_chs
declare -a kernel_sizes
declare -a proj_head_inp_dims
declare -a proj_head_intermediate_dims
declare -a feature_dims


ind=0
for par1 in ${out_ch[@]}
do
    for par2 in ${kernel_size[@]}
    do
        for par3 in ${proj_head_inp_dim[@]}
        do
            for par4 in ${proj_head_intermediate_dim[@]}
            do
                for par5 in ${feature_dim[@]}
                do
                    out_dirs[$ind]=$out_dir"/"${names[0]}$par1"_"${names[1]}$par2"_"${names[2]}$par3"_"${names[3]}$par4"_"${names[4]}$par5
                    out_chs[$ind]=$par1
                    kernel_sizes[$ind]=$par2
                    proj_head_inp_dims[$ind]=$par3
                    proj_head_intermediate_dims[$ind]=$par4
                    feature_dims[$ind]=$par5
                    ((ind=ind+1))
                done
            done
        done
    done
done

out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
out_chs=${out_chs[$SLURM_ARRAY_TASK_ID]}
kernel_sizes=${kernel_sizes[$SLURM_ARRAY_TASK_ID]}
proj_head_inp_dims=${proj_head_inp_dims[$SLURM_ARRAY_TASK_ID]}
proj_head_intermediate_dims=${proj_head_intermediate_dims[$SLURM_ARRAY_TASK_ID]}
feature_dims=${feature_dims[$SLURM_ARRAY_TASK_ID]}

echo out_dirs: $out_dirs
echo out_chs: $out_chs
echo kernel_sizes: $kernel_sizes
echo proj_head_inp_dims: $proj_head_inp_dims
echo proj_head_intermediate_dims: $proj_head_intermediate_dims
echo feature_dims: $feature_dims

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/akitaitsev/anaconda3/lib/
cd /home/akitaitsev/code/intersubject_generalization/dnn/resnet50/EEG/dataset1/

python resnet50-eeg-leftthomas.py -gpu -batch_size $batch_size -out_dir $out_dirs -lr $lr -bpl $bpl -epta $epta -n_epochs $n_epochs -out_ch $out_chs -kernel_size $kernel_sizes -proj_head_inp_dim $proj_head_inp_dims -proj_head_intermediate_dim $proj_head_intermediate_dims -feature_dim $feature_dims -pick_best_net_state
