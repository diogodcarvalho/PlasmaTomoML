## COMPASS SXR NN model

<p align="center">
  <img src=https://github.com/diogodcarvalho/PlasmaTomoML/blob/master/COMPASS/NN/README_examples/COMPASSNet.png width="800">
</p>

The NN was implemented using the Keras library, the building blocks of the NN are found in the following files:
- `nn_model.py` contains the model of the NN
- `nn_callback.py` contains the callbacks used at training time to save the NN parameters
- `nn_train.py` performs the training of the NN

## To prepare the dataset

Different options on which type of dataset we want to use for the training are possible. Choose one of the following deppending on your 
intent.

#### Option 1 

If the geometry you want to apply the NN coincides with the geometry used to generate the training MFR tomograms

 - Run `prepare_data_MFR.py`
   - User defined parameters:
      - `data_directory` path to directory where *.mat files are saved (Default = '../data/Reconstructions/')
      - `ratio` list with percentage of data to training/validation/test set (Default = [.8,.1,.1])
      - `save_path` directory in which all outputs will be saved (Default = './Results/')
   - Outputs:
      - `Results/` directory where all further information will be saved
      - `tomo_COMPASS.npz` contains all tomography relevant information (detector values, tomograms, etc...) and training/validation/test set index division
      - `tomo_GEOM.npz` contains information about camera geometry that is used in these reconstructions

In this case the training/validation/test set belong to the same group of *.mat files generated with the same geometry. No change are made in the detector values used as input of the NN.
    
#### Option 2

If the geometry you want to apply the NN does not coincide with the geometry used to generate the training MFR tomograms

 - Run `prepare_data_MFR_virtual.py` 
   - User defined parameters:
      - `data_directory_old` path to directory where *.mat files for the training/validation set are saved (Default = '../data/Reconstructions/')
      - if `test_set_available = True` assumes there exists directory with reconstruction computed for the new geometry and will use it as test set, new geometry is loaded automatically from the correspondent *.mat files
        - `data_directory_new` path to test set reconstructions 
      - if `test_set_available = False` no test set will be created and user needs to define the new geometry
        - `GEOM` new geometry (ex. 201701)
        - `SXRA_new`,`SXRB_new`,`SXRF_new` used detectors list
      - `ratio` list with percentage of data to training/validation set (Default = [.9,.1])
      - `save_path` directory in which all outputs will be saved (Default = './Results/')
   - Outputs:
      - `Results/` directory where all further information will be saved
      - `tomo_COMPASS.npz` contains all tomography relevant information (detector values, tomograms, etc...) and training/validation/test set index division
      - `tomo_GEOM.npz` contains information about NEW camera geometry
    - Notes:
      - if if `test_set_available = False` the files which depend on the existance of a test set (ex. `calc_metrics.py`, `plot_comparison.py`) cannot be used.
      
In this case the training/validation set belong to a group of reconstructions computed for a camera geometry different to the one we wish to  apply the NN. The original detector values are disregarded and virtual detectors (based on the new geometry) are computed in the old tomograms. These virtual detectors will be the ones used as inputs of the NN. The test set (if it exists) is composed of tomograms obtained for the new geometry.

## To train the NN

- Run `nn_train.py` to train the NN 
  - User defined parameters:
    - `fname` file from which to load the data (Default = '../data/train_data.hdf')
    - `faulty` if True will use faulty detectors, if False will set them to zero (Default = False)
    - `ratio` list with percentage of data to training/validation/test set (Default = [.8,.1,.1])
    - `loss` loss function used (Default = 'mae')
    - `filters` number of convolution filters, changes the size of the dense layers proportionally (Default = 20)
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
    - The training process can be stopped at any point by pressing Ctrl+c
    - This script must be run before any of the others can be used

## To plot the loss function behaviour

- Run `plot_loss.py` to plot the training/validation loss evolution during training 
  - User defined parameters:
    - `save_path` directory in which the 'train.log' is located and where all outputs will be saved (Default = './Results/')
  - Outputs:
    - `loss_log.png` loss function evolution (example below)

<p align="center">
  <img src="https://github.com/diogodcarvalho/PlasmaTomoML/blob/master/JET/NN/README_examples/loss_log.png" width="400"/>
</p> 

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
  <img src=https://github.com/diogodcarvalho/PlasmaTomoML/blob/master/JET/NN/README_examples/JET_92213_49.62_54.02.png width="700"/>
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
  <img src=https://github.com/diogodcarvalho/PlasmaTomoML/blob/master/JET/NN/README_examples/JET_90283.0_46.2981.png width="700"/>
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
 
 
