## To extract the regularization matrix

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
    - This script must be run before any of the others can be used

## To plot the regularization patterns 

- Run `plot_M.py` to create .png files with the regularization patterns obtained by the fitting of M
  - User defined parameters:
    - `save_path` directory where the matrix `M.npy` was stored (Default = './Results/')
    - `simple` if True axis labels, ticks and colorbar are removed
    - `v_min`,`v_max` plot dynamic range (Default = 0.,1.)
  - Outputs:
    - `LOS/` directory with all regularization patterns as .png files, will be stored inside `save_path`
    
## To generate full pulse animations

- Run `anim_full_pulse.py` to generate an .mp4 animation with reconstructions for a given pulse
  - User defined parameters:
    - `pulse` pulse you wish to perform reconstructions
    - `fname` path to .hdf file where the information about the choosen pulse is stored, this file should be created beforehand using `PlasmaTomoML/JET/data/get_bolo.py` which only runs in a JET cluster
    - `save_path` directory where the matrix `M.npy` was stored (Default = './Results/')
    - `plot_los` if True animation with lines of sight appearing is included (Default = 'True')
    - `tmin`,`tmax` animation start and end time (Default uses full pulse)
  - Outputs:
      - `xx.mp4` animation where xx = `pulse`, will be stored inside `save_path`
      - `xx.npz` file containing time vector, bolometer values and correspondent reconstructions, will be stored inside `save_path`
      
<p align="center">
  <img src="https://github.com/diogodcarvalho/PlasmaTomoML/blob/master/JET/M/README_examples/92213.gif" width="600"/>
</p> 

## To generate multiple reconstructions

- Run `plot_reconstruction_multi.py` generate a grid of reconstructions for one one pulse
  - User defined parameters:
       - `pulse` pulse you wish to perform a full pulse reconstruction
       - `fname` path to .hdf file where the information about, this file should be created beforehand using `PlasmaTomoML/JET/data/get_bolo.py` which only runs in a JET cluster
       - `save_path` directory where the matrix `M.npy` was stored (Default = './Results/') 
       - `tmin`,`tmax` first and last reconstruction time
       - `dt` time step between images
       - `nx`,`ny` number of reconstructions plotted per row and collumn, when changing this you might need to edit the inputs of the function `plt.subplots_adjust()` for a better aspect of the final image. Also make sure the dimensions match the number of reconstructions to plot
  - Outputs:
      - `JET_pulse_tmin_tmax.png` , will be stored inside `save_path` (see example below)
      
<p align="center">
  <img src=https://github.com/diogodcarvalho/PlasmaTomoML/blob/master/JET/M/README_examples/JET_92213_49.62_54.02.png width="700"/>
</p> 

 - Run `plot_comparison.py` to generate .png files with differences between original reconstructions and new ones performed with matrix M
    - User defined parameters:
      - `fname` .hdf file on which `M.npy` was fitted (Default = train_data.hdf)
      - `save_path` directory where the matrix `M.npy` was stored (Default = './Results/')
    - Outputs:
      - `COMPARE/` directory with .png files for all the validation set tomograms, will be stored inside `save_path`
    - Notes:
      - Only the validation set is used (defined in `i_divided.npy` saved by running `fit_M.py`)
      - During the plotting the values of the quality metrics (ssim, psnr, nrmse, e_power) are printed in the terminal
      - If you only wish to calculate the quality metrics run `calc_metrics.py`

- Run `calc_metrics.py` to calculate the quality metrics (ssim,psnr,nrmse,e_power) in the validation set
    - User defined parameters:
      - `fname` .hdf file on which `M.npy` was fitted (Default = train_data.hdf)
      - `save_path` directory where the matrix `M.npy` was stored (Default = './Results/')
    - Outputs:
      - (ssim,psnr,nrmse,e_power) average values printed in the terminal
 
 
