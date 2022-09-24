#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=generic_decoding_time_window13-40
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=00:10:00
#SBATCH --qos=prio

cd /home/akitaitsev/code/intersubject_generalization/linear/generic_decoding/
echo plotting generic decoding results for time window 0-40
# tmp_dir
out_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/control/pca/200/time_window0-40/"
res_av=$out_dir"generic_decoding_results_average.pkl"
cor_av=$out_dir"generic_decoding_correlations_average.pkl"
python plots_generic_decoding.py -res $res_av -cor $cor_av -out_dir $out_dir

res_sw=$out_dir"generic_decoding_results_subjectwise.pkl"
cor_sw=$out_dir"generic_decoding_correlations_subjectwise.pkl"
python plots_generic_decoding.py -res $res_sw -cor $cor_sw -out_dir $out_dir

echo plotting generic decoding results for time window 13-40
out_dir="/home/akitaitsev/tmp/generic_decoding/multiviewica/main/srm/200/time_window13-40/"
#out_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/control/pca/200/time_window13-40/"
res_av=$out_dir"generic_decoding_results_average.pkl"
cor_av=$out_dir"generic_decoding_correlations_average.pkl"
python plots_generic_decoding.py -res $res_av -cor $cor_av -out_dir $out_dir

res_sw=$out_dir"generic_decoding_results_subjectwise.pkl"
cor_sw=$out_dir"generic_decoding_correlations_subjectwise.pkl"
python plots_generic_decoding.py -res $res_sw -cor $cor_sw -out_dir $out_dir
