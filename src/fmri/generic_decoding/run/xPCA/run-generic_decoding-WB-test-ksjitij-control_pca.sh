#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=gen_dec_fmri_WB
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=00:10:00
#SBATCH --qos=prio
# N JONS = 60

layer_list=('layer1' 'layer2' 'layer3' 'fc' 'avgpool')
method='PCA'
dim_red='PCA'
x_dim_red='PCA'
comp_list=(10 50 100)
x_n_comp_list=(100 1000)

real_dir="/scratch/akitaitsev/fMRI_Algonautus/intersubject_generalization/"$method"/full_track/"$dim_red
pred_dir="/scratch/akitaitsev/fMRI_Algonautus/regression/kshitij/"$method"/full_track/x_"$x_dim_red
out_dir='/scratch/akitaitsev/fMRI_Algonautus/generic_decoding/kshitij/full_track/'
region="WB"

declare -a out_dirs
declare -a pred_files
declare -a real_files
declare -a dtypes
declare -a layers
declare -a n_comps
declare -a x_n_comps

ind=0
for n_comp in ${comp_list[@]}
do
    for x_n_comp in ${x_n_comp_list[@]}
    do
        for layer in ${layer_list[@]}
        do
            for dtype in "sw" "av"
            do
                n_comps[$ind]=$n_comp
                x_n_comps[$ind]=$x_n_comp
                layers[$ind]=$layer
                pred_files[$ind]=$pred_dir$x_n_comp"/"$method"/"$method$n_comp"/"$layer"_WB_test_pred_"$dtype".pkl"
                real_files[$ind]=$real_dir$n_comp"/WB_shared_test.pkl"
                out_dirs[$ind]=$out_dir"x_"$x_dim_red$x_n_comp"/"$method"/"$dim_red$n_comp"/"
                dtypes[$ind]=$dtype
                ((ind=ind+1))
            done
        done
    done
done

out_dirs=${out_dirs[$SLURM_ARRAY_TASK_ID]}
pred_files=${pred_files[$SLURM_ARRAY_TASK_ID]}
real_files=${real_files[$SLURM_ARRAY_TASK_ID]}
dtypes=${dtypes[$SLURM_ARRAY_TASK_ID]}
layers=${layers[$SLURM_ARRAY_TASK_ID]}
n_comps=${n_comps[$SLURM_ARRAY_TASK_ID]}
x_n_comps=${x_n_comps[$SLURM_ARRAY_TASK_ID]}

echo region: $region
echo x dim red: $x_dim_red
echo x n comp: $x_n_comps
echo method: $method
echo y dim red: $dim_red
echo y n comp: $n_comps
echo dtype: $dtypes
echo layer $layers

echo real_file: $real_files
echo predicted file $pred_files
echo out_dir: $out_dirs

cd /home/akitaitsev/code/intersubject_generalization/fmri/generic_decoding/
python generic_decoding-layerwise.py -real $real_files -pred $pred_files -regr_type $dtypes -out $out_dirs -region $region -layer $layers
