## Intructions

- Run `fit_M.py` to perform the matrix fitting using gradient descent with momentum implemented on Theano 
  - User defined parameters:
    - `fname` file from which to load the data (Default = '../data/train_data.hdf')
    - `faulty` if True will use faulty detectors, if False will set them to zero (Default = False)
    - `ratio` list with percentage of data to training/validation/test set (Default = [.9,.1] no test set)
    - `learning_rate` gradient descent learning rate, tune for better convergence (Default = 0.01)
    - `momentum` momentum parameter, tune for better convergence (Default = 0.9)
    - `epochs` number of gradient descent iterations (Default = 1e5)
    - `save_path` directory in which all outputs will be saved (Default = './Results/')
  - Outputs:
    - `train.png` computational graph, used to check if code is running on GPU
    - `train.log` stores training set loss function error during training
    - `i_divided.npy` indices of training/validation/test set usefull if tests are to be performed later
    - `M.npy` final matrix obtained
  - Notes:
    - The fitting process can be stopped at any point by pressing Ctrl+c, all the files output will still created and saved

- Run `plot_M.py` to create .png files with the regularization patterns obtained by the fitting of M
  - User defined parameters:
    - `save_path` directory where the matrix `M.npy` was stored (Default = './Results/')
    - `simple` if True axis labels, ticks and colorbar are removed
    - `v_min`,`v_max` plot dynamic range (Default = 0.,1.)
  - Outputs:
    - `LOS/LOSxx.png` folder with all regularization patterns as .png files, will be stored inside `save_path`
  - Notes:
    - `fit_m.py` must be runed beforehand to generate the file `M.npy`

- Run `anim_full_pulse.py` to generate an .mp4 animation with reconstructions for a given pulse
  - User defined parameters:
    - `pulse` pulse you wish to perform a full pulse reconstruction
    - `fname` path to .hdf file where the information about, this file should be created beforehand using `PlasmaTomoML/JET/data/get_bolo.py` which only runs in a JET cluster
    - `save_path` directory where the matrix `M.npy` was stored (Default = './Results/')
    - `plot_los` if True animation with lines of sight appearing is included
    - `tmin`,`tmax` animation start and end time
  - Outputs:
      - `xx.mp4` animation where xx = `pulse`, will be stored inside `save_path`
      - `xx.npz` file containing time vector, bolometer values and correspondent reconstructions, will be stored inside `save_path`
 
