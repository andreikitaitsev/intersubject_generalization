#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=hpar_grid-v3
#SBATCH --mail-type=end
#SBATCH --mem=8000
#SBATCH --time=70:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# gridsearch over perceiver with proj head params with lr=1e-4
# grid for hyperparameters is set based on pervious gridsearch
# N JOBS = 4

declare -a out_dirs
# Non variable params
lr=0.0001
batch_size=128
n_epochs=1000
bpl=10
epta=5
out_dir="/scratch/akitaitsev/intersubject_generalization/dnn/perceiver/EEG/leftthomas/projection_head/hpar-grid-ver3/"
names=("perc_latent_array_dim" "perc_num_latent_dim" "perc_latent_heads" "perc_depth" "out_dim_ENC" "out_dim_PH")

# params to make gridsearch over
perc_latent_array_dim=(200)
perc_num_latent_dim=(50)
perc_latent_head=(2 8)
perc_depth=(3 6)
out_dim_ENC=(200) # as if mvica with PCA 200
out_dim_PH=(100) # as if mvica with PCA 100

declare -a perc_latent_array_dims
declare -a perc_num_latent_dims
declare -a perc_latent_heads
declare -a perc_depths
declare -a out_dim_ENCs
declare -a out_dim_PHs



ind=0
for par1 in ${perc_latent_array_dim[@]}
do
    for par2 in ${perc_num_latent_dim[@]}
    do
        for par3 in ${perc_latent_head[@]}
        do
            for par4 in ${perc_depth[@]}
            do
                for par5 in ${out_dim_ENC[@]}
                do
                    for par6 in ${out_dim_PH[@]}
                    do
                        out_dirs[$ind]=$out_dir"/"${names[0]}$par1"_"${names[1]}$par2"_"${names[2]}$par3"_"${names[3]}$par4"_"${names[4]}$par5"_"${names[5]}$par6"/"
                        perc_latent_array_dims[$ind]=$par1
                        perc_num_latent_dims[$ind]=$par2
                        perc_latent_heads[$ind]=$par3
                        perc_depths[$ind]=$par4
                        out_dim_ENCs[$ind]=$par5
                        out_dim_PHs[$ind]=$par6
                        ((ind=ind+1))
                    done
                done
            done
        done
    done
done

perc_latent_array_dims=${perc_latent_array_dims[$SLURM_ARRAY_TASK_ID]}
perc_num_latent_dims=${perc_num_latent_dims[$SLURM_ARRAY_TASK_ID]}
perc_latent_heads=${perc_latent_heads[$SLURM_ARRAY_TASK_ID]}
perc_depths=${perc_depths[$SLURM_ARRAY_TASK_ID]}
out_dim_ENCs=${out_dim_ENCs[$SLURM_ARRAY_TASK_ID]}
out_dim_PHs=${out_dim_PHs[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}

echo perc_latent_array_dims: $perc_latent_array_dims
echo perc_num_latent_dims: $perc_num_latent_dims
echo perc_latent_heads: $perc_latent_heads
echo perc_depths: $perc_depths
echo out_dim_ENCs $out_dim_ENCs
echo out_dim_PHs: $out_dim_PHs
echo out_dir: $out_dirs

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/akitaitsev/anaconda3/lib/
cd /home/akitaitsev/code/intersubject_generalization/dnn/perceiver/EEG/

python perceiver-projection_head-eeg-leftthomas.py -gpu -batch_size $batch_size -out_dir $out_dirs -lr $lr -bpl $bpl -epta $epta -n_epochs $n_epochs -perc_latent_array_dim $perc_latent_array_dims -perc_num_latent_dim $perc_num_latent_dims -perc_latent_heads $perc_latent_heads -perc_depth $perc_depths -out_dim_ENC $out_dim_ENCs -out_dim_PH $out_dim_PHs

