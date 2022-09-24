#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=gen_dec_fmri_WB
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=00:10:00
#SBATCH --qos=prio
# N JONS = 10

layer_list=('layer1' 'layer2' 'layer3' 'fc' 'avgpool')
method='multiviewica'
dim_red='PCA'
ncomp=50
out_dir='/scratch/akitaitsev/fMRI_Algonautus/generic_decoding/test-kshitij/full_track/'$method"/"$dim_red"/"$ncomp"/"
real_file="/scratch/akitaitsev/fMRI_Algonautus/intersubject_generalization/test/full_track/"$method"/"$dim_red"/"$ncomp"/WB_shared_test.pkl"
pred_dir="/scratch/akitaitsev/fMRI_Algonautus/regression/test-kshitij/full_track/"$method"/"$dim_red"/"$ncomp"/"
region=WB


declare -a pred_files
declare -a dtypes
declare -a layers

ind=0
for layer in ${layer_list[@]}:
do
    for dtype in "sw" "av"
    do
        layers[$ind]=$layer
        pred_files[$ind]=$pred_dir"/"$layer"_WB_test_pred_"$dtype".pkl"
        dtypes[$ind]=$dtype
        ((ind=ind+1))
    done
done

pred_files=${pred_files[$SLURM_ARRAY_TASK_ID]}
dtypes=${dtypes[$SLURM_ARRAY_TASK_ID]}
layers=${layers[$SLURM_ARRAY_TASK_ID]}

echo dtype: $dtypes
echo layer $layers

cd /home/akitaitsev/code/intersubject_generalization/fmri/generic_decoding/
python generic_decoding-layerwise.py -real $real_file -pred $pred_files -regr_type $dtypes -out $out_dir -region $region -layer $layers
