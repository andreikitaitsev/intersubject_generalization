!# /usr/bin/env bash

# requirements
pip install torchextractor
pip install nilearn
pip install decord

# download data
download_link='https://www.dropbox.com/s/agxyxntrbwko7t1/participants_data.zip?dl=0'
out_dir='/scratch/akitaitsev/fMRI_Algonautus/raw_data/'
echo $download_link 
wget -P $out_dir -O participants_data.zip -c $download_link 
unzip participants_data.zip
wget -O example.nii -c https://github.com/Neural-Dynamics-of-Visual-Cognition-FUB/Algonauts2021_devkit/raw/main/example.nii
wget -c https://raw.githubusercontent.com/Neural-Dynamics-of-Visual-Cognition-FUB/Algonauts2021_devkit/main/class_names_ImageNet.txt
