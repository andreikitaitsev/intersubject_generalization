from setuptools import find_packages, setup

setup(name='intersubject_generalization',
    version='0.1.0',
    description="Intersubject Generalization",
    author="Andrei Kitaitsev",
    author_email="andre.kit17@gmail.com",
    keywords="EEG, intersubjectgeneralization, Perceiver, transformer, DNN, encoding models, decoding models",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    license = "MIT"
    )
