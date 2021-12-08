#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=gen_dec_fmri_WB
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=00:10:00
#SBATCH --qos=prio
# N JONS = 2

method='multiviewica'
dim_red='PCA'
ncomp=50
out_dir='/scratch/akitaitsev/fMRI_Algonautus/generic_decoding/test/full_track/'$method"/"$dim_red"/"$ncomp"/"
real_file="/scratch/akitaitsev/fMRI_Algonautus/intersubject_generalization/test/full_track/"$method"/"$dim_red"/"$ncomp"/WB_shared_test.pkl"
pred_dir="/scratch/akitaitsev/fMRI_Algonautus/regression/test/full_track/"$method"/"$dim_red"/"$ncomp"/"
region=WB


declare -a pred_files
declare -a dtypes

ind=0
for dtype in "sw" "av"
do
    pred_files[$ind]=$pred_dir"/WB_test_pred_"$dtype".pkl"
    dtypes[$ind]=$dtype
    ((ind=ind+1))
done

pred_files=${pred_files[$SLURM_ARRAY_TASK_ID]}
dtypes=${dtypes[$SLURM_ARRAY_TASK_ID]}

echo dtype: $dtypes

cd /home/akitaitsev/code/intersubject_generalization/fmri/generic_decoding/
python generic_decoding.py -real $real_file -pred $pred_files -regr_type $dtypes -out $out_dir -region $region
