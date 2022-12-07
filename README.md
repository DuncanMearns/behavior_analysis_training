Training for behavioral analysis

# Data

The data from Mearns et al. (2020) is available on Mendeley [here](https://data.mendeley.com/datasets/mw2mmpdz3g/1)
(total uncompressed size is ~25 GB).

After extracting the data, the directory tree should be organized as follows:
```text
|-- data_directory {call this whatever you want}
    |-- kinematics
        |-- 2017081001
            |-- 2017081001171128.csv
            |-- 2017081001171228.csv
            ...
        |-- 2017081002
        ...
    |-- transitions
        |-- T.npy
        |-- transition_matrices.npy
        |-- USVa.npy
        |-- USVs.npy
        |-- WTW.npy
    |-- exemplar_distance_matrix.npy
    |-- exemplars.csv
    |-- isomap.npy
    |-- mapped_bouts.csv
```

# Environment

To install the environment run:
```commandline
conda env create -f environment.yml 
```

To activate the environment run:
```commandline
conda activate behavior_analysis_training 
```

To install the [ethomap](https://github.com/DuncanMearns/ethomap) package run:
```commandline
pip install ethomap
```


# Jupyter

Install JupyterLab in your base anaconda environment:
```commandline
conda install -c conda-forge jupyterlab
```

To launch JupyterLab:
```commandline
jupyter-lab
```

If this does not work and you get a 404 error try:
```commandline
jupyter serverextension enable --py jupyterlab --user
```
```commandline
conda install -c conda-forge nodejs
```

To make your environment accessible in JupyterLab, first activate your environment (see above) and run:
```commandline
conda install -c anaconda ipykernel
```
```commandline
python -m ipykernel install --user --name=behavior_analysis_training
```
