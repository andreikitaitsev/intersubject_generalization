from pathlib import Path
import zipfile

from src.preprocessig.preprocesor import Preprocessor

# dirs
base_dir = Path.cwd().joinpath("data/")
inp_dir = base_dir.joinpath('preprocessed_data_unzip/')

# preprocess data
preprsessor = Preprocessor(inp_dir)
featuremat_train, featurema_test = preprsessor.preprocess()
