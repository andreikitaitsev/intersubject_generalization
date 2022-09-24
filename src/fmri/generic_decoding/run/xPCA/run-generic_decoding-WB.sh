#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=gen_dec_fmri_WB
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=00:10:00
#SBATCH --qos=prio
# N JONS = 18

region='WB'
track='full_track'
method='multiviewica'
x_dim_red='PCA'
y_dim_red='PCA'
x_n_comp_list=(100 500 1000)
comp_list=(10 50 100)

real_base='/scratch/akitaitsev/fMRI_Algonautus/intersubject_generalization/'$method'/'$track'/'
pred_base='/scratch/akitaitsev/fMRI_Algonautus/regression/'$method'/'$track'/'
out_base='/scratch/akitaitsev/fMRI_Algonautus/generic_decoding/'$track'/'


declare -a out_dirs
declare -a real_files
declare -a pred_files
declare -a dtypes
declare -a n_comps
declare -a x_n_comps

ind=0
for dtype in "sw" "av"
do
    for n_comp in ${comp_list[@]}
    do
        for x_n_comp in ${x_n_comp_list[@]}
        do
            out_dirs[$ind]=$out_base'/x_'$x_dim_red$x_n_comp'/'$method'/'$y_dim_red$n_comp'/'
            real_files[$ind]=$real_base'/'$y_dim_red$n_comp'/'$region'_shared_test.pkl'
            pred_files[$ind]=$pred_base'/x_'$x_dim_red$x_n_comp'/'$method'/'$y_dim_red$n_comp'/'$region'_test_pred_'$dtype'.pkl'
            dtypes[$ind]=$dtype
            n_comps[$ind]=$n_comp
            x_n_comps[$ind]=$x_n_comp
            ((ind=ind+1))
        done
   done
done

out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
real_files=${real_files[$SLURM_ARRAY_TASK_ID]}
pred_files=${pred_files[$SLURM_ARRAY_TASK_ID]}
dtypes=${dtypes[$SLURM_ARRAY_TASK_ID]}
n_comps=${n_comps[$SLURM_ARRAY_TASK_ID]}
x_n_comps=${x_n_comps[$SLURM_ARRAY_TASK_ID]}

echo region: $region
echo real_file: $real_files
echo pred_file: $pred_files
echo output_dir: $out_dirs
echo n_comps: $n_comps
echo x_n_comps: $x_n_comps
echo dtype: $dtypes

cd /home/akitaitsev/code/intersubject_generalization/fmri/generic_decoding/
echo Running generic decoding on $track data with x $x_dim_red and y $method $n_comps .
python generic_decoding.py -real $real_files -pred $pred_files -regr_type $dtypes -out $out_dirs -region $region
