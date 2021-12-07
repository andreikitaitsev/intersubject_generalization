#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=gen_dec_fmri_WB
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=00:10:00
#SBATCH --qos=prio
# N JONS = 36

out_base='/scratch/akitaitsev/fMRI_Algonautus/generic_decoding/mini_track/'
real_base="/scratch/akitaitsev/fMRI_Algonautus/intersubject_generalization/mini_track/"
pred_base="/scratch/akitaitsev/fMRI_Algonautus/regression/mini_track/"

region_list=('EBA' 'FFA' 'LOC' 'PPA' 'STS' 'V1' 'V2' 'V3' 'V4')
comp_list=(10 50)

declare -a out_dirs
declare -a real_files
declare -a pred_files
declare -a dtypes
declare -a n_comps
declare -a regions

ind=0
for dtype in "sw" "av"
do
    for n_comp in ${comp_list[@]}
    do
        for region in ${region_list[@]}
        do
            out_dirs[$ind]=$out_base"/PCA"$n_comp"/"$region"/"
            real_files[$ind]=$real_base"/PCA"$n_comp"/"$region"_shared_test.pkl"
            pred_files[$ind]=$pred_base"/PCA"$n_comp"/"$region/"Y_test_pred_"$dtype".pkl"
            dtypes[$ind]=$dtype
            n_comps[$ind]=$n_comp
            regions[$ind]=$region
            ((ind=ind+1))
        done
   done
done

out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
real_files=${real_files[$SLURM_ARRAY_TASK_ID]}
pred_files=${pred_files[$SLURM_ARRAY_TASK_ID]}
dtypes=${dtypes[$SLURM_ARRAY_TASK_ID]}
n_comps=${n_comps[$SLURM_ARRAY_TASK_ID]}

echo output_dir: $out_dirs
echo n_comps: $n_comps
echo dtype: $dtypes

cd /home/akitaitsev/code/intersubject_generalization/fmri/generic_decoding/
echo Running generic decoding on control data with pca=200
python generic_decoding.py -real $real_files -pred $pred_files -regr_type $dtypes -out $out_dirs
