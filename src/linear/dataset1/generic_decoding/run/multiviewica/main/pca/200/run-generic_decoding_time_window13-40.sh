#! /bin/bash
#SBATCH --mail-user=akitaitsev@zedat.fu-berlin.de
#SBATCH --job-name=generic_decoding_time_window13-40
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=00:10:00
#SBATCH --qos=prio


out_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/on_generalized_data/time_window13-40/pca_200/"
real="/scratch/akitaitsev/intersubject_generalizeation/linear/multiviewica/time_window13-40/pca_200/shared_test.pkl"
cd /home/akitaitsev/code/intersubject_generalization/linear/generic_decoding/

echo Running generic decoding for average data for time window 13-40.
pred_av="/scratch/akitaitsev/intersubject_generalizeation/linear/regression/on_generalized_data/multiviewica/time_window13-40/pca_200/shared_test_predicted_average.pkl"
python generic_decoding.py -real $real -pred $pred_av -d_type "average" -out $out_dir 


echo running generic decoding for subjectwise data for time window 13-40.
pred_sw="/scratch/akitaitsev/intersubject_generalizeation/linear/regression/on_generalized_data/multiviewica/time_window13-40/pca_200/shared_test_predicted_subjectwise.pkl"
python generic_decoding.py -real $real -pred $pred_sw -d_type "subjectwise" -out $out_dir 


echo plotting generic decoding results for averaged data
res_av=$out_dir"generic_decoding_results_average.pkl"
cor_av=$out_dir"generic_decoding_correlations_average.pkl"
python plots_generic_decoding.py -res $res_av -cor $cor_av -out_dir $out_dir

echo plotting generic decoding results for subjectwise data 
res_sw=$out_dir"generic_decoding_results_subjectwise.pkl"
cor_sw=$out_dir"generic_decoding_correlations_subjectwise.pkl"
python plots_generic_decoding.py -res $res_sw -cor $cor_sw -out_dir $out_dir
