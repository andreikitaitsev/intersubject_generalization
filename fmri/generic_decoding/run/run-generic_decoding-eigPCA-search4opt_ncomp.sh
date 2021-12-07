#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=gen_dec_fmri_WB
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=00:10:00
#SBATCH --qos=prio
# N JONS = 20

out_base='/scratch/akitaitsev/fMRI_Algonautus/generic_decoding/comp_search/'
real_base="/scratch/akitaitsev/fMRI_Algonautus/intersubject_generalization/comp_search/"
pred_base="/scratch/akitaitsev/fMRI_Algonautus/regression/comp_search/"
method="multiviewica"
dim_red="eig_PCA"

region_list=('WB' 'EBA' 'FFA' 'LOC' 'PPA' 'STS' 'V1' 'V2' 'V3' 'V4')
comp_list=(10 50 200 800)

declare -a out_dirs
declare -a real_dir
declare -a pred_dir
declare -a dtypes
declare -a n_comps
declare -a regions

ind=0
for dtype in "sw" "av"
do
    for region in ${region_list[@]}
    do
        out_dirs[$ind]=$out_base"/"$method"/"$dim_red"/"
        real_dir[$ind]=$real_base"/"$method"/"$dim_red"/"
        pred_dir[$ind]=$pred_base"/"$method"/"$dim_red"/"
        dtypes[$ind]=$dtype
        regions[$ind]=$region
        ((ind=ind+1))
    done
done

regions=${regions[$SLURM_ARRAY_TASK_ID]}
out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
real_dir=${real_dir[$SLURM_ARRAY_TASK_ID]}
pred_dir=${pred_dir[$SLURM_ARRAY_TASK_ID]}
dtypes=${dtypes[$SLURM_ARRAY_TASK_ID]}

echo region: $regions
echo output_dir: $out_dirs
echo dtype: $dtypes

cd /home/akitaitsev/code/intersubject_generalization/fmri/generic_decoding/
python generic_decoding.py -real $real_dir -pred $pred_dir -regr_type $dtypes -out $out_dirs -region $regions -auto_scan
