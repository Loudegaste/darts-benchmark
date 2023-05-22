from setuptools import setup, find_packages

setup(
    name='darts_benchmark',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'pytorch-lightning',
        'darts==0.24.0',
        'optuna~=3.1',
        'ray~=2.3',
    ],
)