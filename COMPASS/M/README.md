## To prepare the dataset
 - Run `prepare_data_MFR.py` to read the *.mat files generated by MFR Matlab implementation and prepare the dataset for matrix fitting
   - User defined parameters:
      - `data_directory` path to directory where *.mat files are saved (Default = '../data/Reconstructions/')
      - `ratio` list with percentage of data to training/validation/test set (Default = [.9,.1] no test set)
      - `save_path` directory in which all outputs will be saved (Default = './Results/')
   - Outputs:
      - `Results/` directory where all further information will be saved
      - `tomo_COMPASS.npz` contains all tomography relevant information (detector values, tomograms, etc...) and training/validation set index division
      - `tomo_GEOM.npz` contains information about camera geometry that is used in these reconstructions
 - Notes:
    - This script must be run before any of the others can be used
    
## To extract the regularization matrix

- Run `fit_M.py` to perform the matrix fitting using gradient descent with momentum implemented on Theano
  - User defined parameters:
    - `save_path` directory where `tomo_COMPASS.npz` was saved, all outputs will also be stored here (Default = './Results/')
    - `learning_rate` gradient descent learning rate, tune for better convergence (Default = 1.0)
    - `momentum` momentum parameter, tune for better convergence (Default = 0.9)
    - `epochs` number of gradient descent iterations (Default = 1e5)
  - Outputs:
    - `train.png` computational graph, used to check if code is running on GPU
    - `train.log` stores training set loss function error during training
    - `M.npy` final matrix obtained
  - Notes:
    - The fitting process can be stopped at any point by pressing Ctrl+c, all the files output will still created and saved

## To plot the regularization patterns 

- Run `plot_M.py` to create .png files with the regularization patterns obtained by the fitting of M
  - User defined parameters:
    - `save_path` directory where the matrix `M.npy` was stored (Default = './Results/')
    - `simple` if True axis labels, ticks and colorbar are removed
    - `v_min`,`v_max` plot dynamic range (Default = 0.,1.)
  - Outputs:
    - `LOS/` directory with all regularization patterns as .png files, is located inside `save_path`

<p align="center">
  <img src=https://github.com/diogodcarvalho/PlasmaTomoML/blob/master/JET/M/README_examples/JET_LOS_4.png width="200" />
  <img src=https://github.com/diogodcarvalho/PlasmaTomoML/blob/master/JET/M/README_examples/JET_LOS_9.png width="200" /> 
  <img src=https://github.com/diogodcarvalho/PlasmaTomoML/blob/master/JET/M/README_examples/JET_LOS_14.png width="200" />
  <img src=https://github.com/diogodcarvalho/PlasmaTomoML/blob/master/JET/M/README_examples/JET_LOS_21.png width="200" />
</p>

## To calculate quality metrics

- Run `calc_metrics.py` to calculate the quality metrics in the validation set
    - User defined parameters:
      - `save_path` directory where the matrix `M.npy` was stored (Default = './Results/')
    - Outputs:
      - `METRICS/` directory where all outputs are saved, is located inside `save_path`
      - `metrics.csv` contains all metrics calculated for specific pulse and time-step
      - `AMRE.png` average absolute mean error pixelwise
      - `MRE.png` average mean relative error pixelwise
      - `NRMSE.png` average normalise root mean squared error pixelwise
    - Notes:
      - Average metrics values are printed in the terminal

## To compare original and new reconstructions

 - Run `plot_comparison.py` to generate .png files with differences between original reconstructions and new ones performed with matrix M
    - User defined parameters:
      - `save_path` directory where the matrix `M.npy` was stored (Default = './Results/')
      - `pulses` list of pulses to plot
    - Outputs:
      - `COMPARE/` directory with .png files for all the validation set tomograms, will be stored inside `save_path`
    - Notes:
      - Only the validation set is used
      - If you only wish to calculate the quality metrics run `calc_metrics.py`

<p align="center">
  <img src=https://github.com/diogodcarvalho/PlasmaTomoML/blob/master/JET/M/README_examples/JET_89077_48.6.png width="700"/>
</p> 
 
 