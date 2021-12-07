#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=fmri-extract_features
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=30:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# N JOBS = 6

video_dir='/scratch/akitaitsev/fMRI_Algonautus/raw_data/AlgonautsVideos268_All_30fpsmax/'
out_dir_base='/scratch/akitaitsev/fMRI_Algonautus/activations/cornet_s/trial2/'
comp_list=(10 50 100 200 400 800)

declare -a n_comps
declare -a out_dirs
ind=0
for n_comp in ${comp_list[@]}
do
    n_comps[$ind]=$n_comp
    out_dirs[$ind]=$out_dir_base'PCA'$n_comp'/'
    ((ind=ind+1))
done

n_comps=${n_comps[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}

echo n_comp: $n_comps
echo out_dir: $out_dirs

# default preprocessor is PCA
cd /home/akitaitsev/code/intersubject_generalization/fmri/
python extract_features.py -video_dir $video_dir -out_dir $out_dirs -n_comp $n_comps
