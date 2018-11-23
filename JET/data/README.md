## To download tomograms + bolometer measures from JET database

- Run `get_tomo_JET_database.py` 
  - User defined parameters:
    - `pulses` pulse range from which tomograms should be downloaded
    - `fname` name of file where data will be saved (Default = 'tomo_JET.hdf')
  - Outputs:
    - `<fname>.hdf` data file with tomograms and bolometer measures
    

##  To download solely bolometer measures

- Run `get_bolo_JET_database.py` 
  - User defined parameters:
    - `pulses` pulse range from which tomograms should be downloaded
    - `fname` name of file where data will be saved (Default = 'bolo_JET.hdf')
  - Outputs:
    - `<fname>.hdf` data file with tomograms and bolometer measures
    
