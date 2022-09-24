#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=fmri-extract_features
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=03:00:00
#SBATCH --qos=prio
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# N JOBS = 1

n_comp=100
video_dir='/scratch/akitaitsev/fMRI_Algonautus/raw_data/AlgonautsVideos268_All_30fpsmax/'
act_dir='/scratch/akitaitsev/fMRI_Algonautus/dnn_features/resnet18/kshitij/activations/'
pca_dir='/scratch/akitaitsev/fMRI_Algonautus/dnn_features/resnet18/kshitij/pca'$n_comp'/'


echo Running feature extraction with cornet-s DNN with PCA downsampling...
echo n_comp: $n_comp
echo act_dir: $act_dir
echo pca dir: $pca_dir

# default preprocessor is PCA
cd /home/akitaitsev/code/intersubject_generalization/fmri/extract_features/
python extract_features-resnet18-kshitij.py -video_dir $video_dir -act_dir $act_dir -pca_dir $pca_dir -n_comp $n_comp
