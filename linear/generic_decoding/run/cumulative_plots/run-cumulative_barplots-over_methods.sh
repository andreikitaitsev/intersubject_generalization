#! /bin/bash
# better run with srun as it takes couple of seconds
cd /home/akitaitsev/code/intersubject_generalization/linear/generic_decoding 
#0-40
#out_dir="/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/cumulative/"
out_dir="/home/akitaitsev/"
control="/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/control/pca/200/time_window0-40/generic_decoding_results_subjectwise.pkl"
subjwise="/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/multiviewica/main/pca/200/time_window0-40/generic_decoding_results_subjectwise.pkl"
average="/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/multiviewica/main/pca/200/time_window0-40/generic_decoding_results_average.pkl"

python cumulative_barplots.py -ctrl $control -sw $subjwise -av $average -out $out_dir
# 0-13
control="/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/control/pca/200/time_window13-40/generic_decoding_results_subjectwise.pkl"
subjwise="/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/multiviewica/main/pca/200/time_window13-40/generic_decoding_results_subjectwise.pkl"
average="/scratch/akitaitsev/intersubject_generalizeation/linear/generic_decoding/multiviewica/main/pca/200/time_window13-40/generic_decoding_results_average.pkl"
python cumulative_barplots.py -ctrl $control -sw $subjwise -av $average -out $out_dir

