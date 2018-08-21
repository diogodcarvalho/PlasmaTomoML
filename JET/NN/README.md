## JET Bolometer NN model

<p align="center">
  <img src=https://github.com/diogodcarvalho/PlasmaTomoML/blob/master/JET/NN/README_examples/JETNet.png width="600">
</p>

The NN was implemented using the Keras library, the building blocks of the NN are found in the following files:
- `nn_model.py` contains the model of the NN
- `nn_callback.py` contains the callbacks used at training time to save the NN parameters
- `nn_train.py` performs the training of the NN



## To train the NN

- Run `nn_train.py` to train the NN 
  - User defined parameters:
    - `fname` file from which to load the data (Default = '../data/train_data.hdf')
    - `faulty` if True will use faulty detectors, if False will set them to zero (Default = False)
    - `ratio` list with percentage of data to training/validation/test set (Default = [.8,.1,.1])
    - `learning_rate` gradient descent learning rate, tune for better convergence (Default = 1e-4)
    - `epochs` number of gradient descent iterations (Default = 1e5)
    - `batch_size` (Default = 398)
    - `save_path` directory in which all outputs will be saved (Default = './Results/')
  - Outputs:
    - `model_options.log` file with selected training option info (filters,lr,epochs,...)
    - `train.log` stores training and validation loss function error during training
    - `i_divided.npy` indices of training/validation/test set usefull if tests are to be performed later
    - `model_parameters.hdf` best NN parameters obtained
  - Notes:
    - The fitting process can be stopped at any point by pressing Ctrl+c, all the files output will still created and saved
    - This script must be run before any of the others can be used

## To generate full pulse animations

- Run `anim_full_pulse.py` to generate an .mp4 animation with reconstructions for a given pulse
  - User defined parameters:
    - `pulse` pulse you wish to perform reconstructions
    - `fname` path to .hdf file where the information about the choosen pulse is stored, this file should be created beforehand using `PlasmaTomoML/JET/data/get_bolo.py` which only runs in a JET cluster
    - `save_path` directory where the NN parameters `model_parameters.hdf` was stored (Default = './Results/')
    - `plot_los` if True animation with lines of sight appearing is included (Default = 'True')
    - `tmin`,`tmax` animation start and end time (Default uses full pulse)
  - Outputs:
      - `xx.mp4` animation where xx = `pulse`, will be stored inside `save_path`
      - `xx.npz` file containing time vector, bolometer values and correspondent reconstructions, will be stored inside `save_path`
      
<p align="center">
  <img src="https://github.com/diogodcarvalho/PlasmaTomoML/blob/master/JET/NN/README_examples/92213.gif" width="600"/>
</p> 

## To generate multiple reconstructions

- Run `plot_reconstruction_multi.py` generate a grid of reconstructions for one one pulse
  - User defined parameters:
       - `pulse` pulse you wish to perform a full pulse reconstruction
       - `fname` path to .hdf file where the information about, this file should be created beforehand using `PlasmaTomoML/JET/data/get_bolo.py` which only runs in a JET cluster
       - `save_path` directory where the NN parameters `model_parameters.hdf` was stored (Default = './Results/')
       - `tmin`,`tmax` first and last reconstruction time
       - `dt` time step between images
       - `nx`,`ny` number of reconstructions plotted per row and collumn, when changing this you might need to edit the inputs of the function `plt.subplots_adjust()` for a better aspect of the final image. Also make sure the dimensions match the number of reconstructions to plot
  - Outputs:
      - `JET_pulse_tmin_tmax.png` , will be stored inside `save_path` (see example below)
      
<p align="center">
  <img src=https://github.com/diogodcarvalho/PlasmaTomoML/blob/master/JET/M/README_examples/JET_92213_49.62_54.02.png width="700"/>
</p> 

## To compare original and new reconstructions

 - Run `plot_comparison.py` to generate .png files with differences between original reconstructions and new ones performed with matrix M
    - User defined parameters:
      - `fname` .hdf file on which `M.npy` was fitted (Default = train_data.hdf)
      - `save_path` directory where the NN parameters `model_parameters.hdf` was stored (Default = './Results/')
    - Outputs:
      - `COMPARE/` directory with .png files for all the validation set tomograms, will be stored inside `save_path`
    - Notes:
      - Only the validation set is used (defined in `i_divided.npy` saved by running `nn_train.py`)
      - During the plotting the values of the quality metrics (ssim, psnr, nrmse, e_power) are printed in the terminal
      - If you only wish to calculate the quality metrics run `calc_metrics.py`

<p align="center">
  <img src=https://github.com/diogodcarvalho/PlasmaTomoML/blob/master/JET/NN/README_examples/JET_89077_48.6.png width="700"/>
</p> 

## To calculate quality metrics

- Run `calc_metrics.py` to calculate the quality metrics (ssim,psnr,nrmse,e_power) in the validation set
    - User defined parameters:
      - `fname` .hdf file on which the NN was fitted (Default = train_data.hdf)
      - `save_path` directory where the NN parameters `model_parameters.hdf` was stored (Default = './Results/')
    - Outputs:
      - (ssim,psnr,nrmse,e_power) average values printed in the terminal   

## To calculate quality metrics with shut-down detectors

- Run `calc_metrics_dropout.py` to calculate the quality metrics (ssim,psnr,nrmse,e_power) in the validation set after shutting down a given number of detectors at a time. Performs calculations on all possible combinations and outputs average value 
    - User defined parameters:
      - `fname` .hdf file on which the NN was fitted (Default = train_data.hdf)
      - `save_path` directory where the NN parameters `model_parameters.hdf` was stored (Default = './Results/')
      - `n_shutdown` number of detectors to shut-down
    - Outputs:
      - (ssim,psnr,nrmse,e_power) average values printed in the terminal
 
 
