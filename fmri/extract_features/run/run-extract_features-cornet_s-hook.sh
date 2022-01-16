#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=fmri-extract_features
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=03:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# N JOBS = 3

video_dir='/scratch/akitaitsev/fMRI_Algonautus/raw_data/AlgonautsVideos268_All_30fpsmax/'
out_dir_base='/scratch/akitaitsev/fMRI_Algonautus/dnn_features/cornet_s/'
comp_list=(100 500 1000)

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

echo Running feature extraction with cornet-s DNN with PCA downsampling...
echo n_comp: $n_comps
echo out_dir: $out_dirs

# default preprocessor is PCA
cd /home/akitaitsev/code/intersubject_generalization/fmri/extract_features/
python extract_features-cornet_s.py -video_dir $video_dir -out_dir $out_dirs -n_comp $n_comps
