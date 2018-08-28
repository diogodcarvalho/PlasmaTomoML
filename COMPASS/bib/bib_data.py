
import glob
import scipy.io
import numpy as np

def get_tomo_MFR(data_directory, subsampling = 1, flatten = False, verbose= False):
	"""
	Reads all *.mat files in data_directory, which contain the reconstructions
	performed with the original MFR algorithm implemented in Matlab, and 
	returns the data present in those files. 
	Only reconstructions with |chi2-1|<0.05 are kept.

	Inputs:
		data_directory - path directory where reconstructions are saved
		subsampling - (int) ex. 4 means store at every 4th reconstruction (default is no subsampling)
		flatten - (bool) if True flatten tomography image to 1D array
		verbose - (bool) if True writes read file information
	Outputs:
		sxr - measurement from SXR cameras (includes non active) [kW/m^-2]
		tomo - reconstruction [kW/m^-3]
		esxr - expected error from SXR cameras
		sxrfit - virtual bolometers measures [kW/m^-2]
		t - time corresponding to time
		chi2 - chi2 values
		pulses - list of pulses
		SXRA - detectors used from camera A
		SXRB - "" B
		SXRF - "" F
	"""

	t = []
	tomo = []
	sxr = []
	esxr = []
	chi2 = []
	sxrfit = []
	pulses = []

	print 'Reading directory: ', data_directory

	# ensure files are opened by pulse number order 
	filenames_dict = {int(filename[filename.find('shot')+len('shot'):filename.find('_reg')]):filename for filename in glob.glob(data_directory + '*.mat')}
	filenames_pulses = list(filenames_dict.keys())
	filenames_pulses.sort()

	for pulse in filenames_pulses:

		filename = filenames_dict[pulse]

		if verbose:
			print filename

		f = scipy.io.loadmat(filename)

		#------------------------------------------------------------------ 
		# COMPASS Matlab File dictionary keys meaning
		#
		# G_final	- reconstructions (#time_steps, #pixels)
		# tvec		- time vector 
		# CHI2      - chi^2 values (~1)
		# Y 		- input signal from detectors (#time_steps, #detectors)
		# Yrfit 	- virtual signals from detectors after reconstruction
		# dets 		- list of detectors used
		# dY 		- expected error of detectors
		#------------------------------------------------------------------

		t_pulse = f['tvec'][:][0]
		tomo_pulse =  np.swapaxes(np.swapaxes(f['G_final'][:],0,2),1,2)
		sxr_pulse = f['Y'][:]
		esxr_pulse = f['dY'][:]
		chi2_pulse = f['CHI2'][:,0]
		sxrfit_pulse = f['Yrfit'][:]

		if verbose:
			print pulse, t_pulse.shape, t_pulse.dtype, tomo_pulse.shape, tomo_pulse.dtype, sxr_pulse.shape, sxr_pulse.dtype, chi2_pulse.shape,\
			esxr_pulse.shape, esxr_pulse.dtype

		# remove bad reconstructions
		index = abs(chi2_pulse-1) < 0.05
		t_pulse = t_pulse[index]
		tomo_pulse = tomo_pulse[index,:,:]
		sxr_pulse = sxr_pulse[index,:]
		esxr_pulse = esxr_pulse[index,:]
		sxrfit_pulse = sxrfit_pulse[index,:]
		chi2_pulse = chi2_pulse[index]

		# subsampling
		t_pulse = t_pulse[::subsampling]
		tomo_pulse = tomo_pulse[::subsampling,:,:]
		sxr_pulse = sxr_pulse[::subsampling,:]
		esxr_pulse = esxr_pulse[::subsampling,:]
		sxrfit_pulse = sxrfit_pulse[::subsampling,:]
		chi2_pulse = chi2_pulse[::subsampling]

		# flatten
		if flatten:
			tomo_pulse = tomo_pulse.reshape(tomo_pulse.shape[0],-1)

		if verbose:
			print pulse, t_pulse.shape, t_pulse.dtype, tomo_pulse.shape, tomo_pulse.dtype, sxr_pulse.shape, sxr_pulse.dtype, esxr_pulse.shape, esxr_pulse.dtype

		for i in range(len(t_pulse)):
			t.append(t_pulse[i])
			tomo.append(tomo_pulse[i])
			sxr.append(sxr_pulse[i])
			esxr.append(esxr_pulse[i])
			sxrfit.append(sxrfit_pulse[i])
			chi2.append(chi2_pulse[i])
			pulses.append(int(pulse))

	t = np.asarray(t, dtype = np.float32)
	tomo = np.asarray(tomo, dtype = np.float32)/1e3
	sxr = np.asarray(sxr, dtype = np.float32)/1e3
	esxr = np.asarray(esxr, dtype = np.float32)/1e3
	sxrfit = np.asarray(sxrfit, dtype = np.float32)/1e3
	chi2 = np.asarray(chi2, dtype = np.float32)
	pulses = np.asarray(pulses)

	if verbose:
		print pulses.shape, t.shape, tomo.shape, sxr.shape, esxr.shape, sxrfit.shape, pulses.shape

	# save detectors and last pulse used. Later it will be needed to know the geometry
	# so the chi2 value can be correctly calculated 
	# -1 due to conversion from matlab to python indeces
	SXRA = np.squeeze(f['dets'][0][0]) - 1
	SXRB = np.squeeze(f['dets'][0][1]) - 1
	SXRF = np.squeeze(f['dets'][0][2]) - 1
	

	return sxr, tomo, esxr, sxrfit,t,chi2,pulses,SXRA,SXRB,SXRF
