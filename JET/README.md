## Directory structure

- `M/`  code necessary to extract the regularization matrix from existing reconstructions and perform new ones
- `NN/` code neccessary to train and apply the NN for JET bolometer system
- `bib/`  libraries created to perform a broad set of tasks common to the `NN/` and `M/` folders
- `data/` contains scripts to download the bolometer measurements and correspondent reconstructions from JET database, will only work inside a JET cluster. The downloaded data files will be stored here.
- `geom/` information about the JET vessel geometry and lines of sight of the kb5 bolomter system

Further instructions on how to work with the code is given inside the `M/` and `NN/` directories depending on which method you pretend to apply

## IMPORTANT - Accessing JET data

Before any NN is trained or regularization is fitted one has to create *.hdf file with the tomograms present in JET database. To do so go to the `data/` directory and run `get_tomo_JET_database.py`. The file needs to be edited to choose the desired pulses and will only work inside a JET cluster. All data files created will be stored in this directory

