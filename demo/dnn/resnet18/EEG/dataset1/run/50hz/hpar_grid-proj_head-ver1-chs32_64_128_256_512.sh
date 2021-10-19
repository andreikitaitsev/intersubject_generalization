#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=hpar_grid-v4
#SBATCH --mail-type=end
#SBATCH --mem=8000
#SBATCH --time=70:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# N JOBS = 6

srate=50
lr=0.0001
batch_size=128
n_epochs=600
bpl=10
epta=5
out_dir="/scratch/akitaitsev/intersubject_generalization/dnn/resnet18/EEG/dataset1/"$srate"hz/hpar-grid-ver1/"
chs=(32 64 128 256 512)
names=("chs32_64_128_256_512" "proj_head_inp_dim" "proj_head_intermediate_dim" "n_features")


pr_head_inp_dim=(512 1024)
pr_head_int_dim=(1024)
n_feature=(200 400 1000)


declare -a out_dirs
declare -a pr_head_inp_dims
declare -a pr_head_int_dims
declare -a n_features

ind=0
for par1 in ${pr_head_inp_dim[@]}
do
    for par2 in ${pr_head_int_dim[@]}
    do
        for par3 in ${n_feature[@]}
        do
            out_dirs[$ind]=$out_dir"/"${names[0]}"_"${names[1]}$par1"_"${names[2]}$par2"_"${names[3]}$par3
            pr_head_inp_dims[$ind]=$par1
            pr_head_int_dims[$ind]=$par2
            n_features[$ind]=$par3
            ((ind=ind+1))
        done
    done
done

out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
pr_head_inp_dims=${pr_head_inp_dims[$SLURM_ARRAY_TASK_ID]}
pr_head_int_dims=${pr_head_int_dims[$SLURM_ARRAY_TASK_ID]}
n_features=${n_features[$SLURM_ARRAY_TASK_ID]}

echo out_dir: $out_dirs
echo chs: ${chs[@]}
echo pr_head_inp_dims: $pr_head_inp_dims
echo pr_head_int_dims: $pr_head_int_dims
echo n_features: $n_features

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/akitaitsev/anaconda3/lib/
cd /home/akitaitsev/code/intersubject_generalization/dnn/resnet18/EEG/

python resnet18-eeg.py -gpu -batch_size $batch_size -out_dir $out_dirs -lr $lr -bpl $bpl -epta $epta -n_epochs $n_epochs -proj_head_inp_dim $pr_head_inp_dims -proj_head_intermediate_dim $pr_head_int_dims -feature_dim $n_features -out_chs_conv1 ${chs[0]} -out_chs1 ${chs[1]} -out_chs2 ${chs[2]} -out_chs3 ${chs[3]} -out_chs4 ${chs[4]}
