from pathlib import Path
import zipfile


# data
subjects = ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')
subjects = ['sub-{}'.format(el) for el in subjects]
        
base_dir = Path.cwd().joinpath("data/")
inp_dir = base_dir.joinpath('preprocessed_data')
out_dir = base_dir.joinpath('preprocessed_data_unzip')
if not out_dir.is_dir():
    out_dir.mkdir()

for subj in subjects:
    with zipfile.ZipFile(inp_dir.joinpath('{}.zip'.format(subj)), 'r') as zip_ref:
        zip_ref.extractall(out_dir)
