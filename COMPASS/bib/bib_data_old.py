
import os
import h5py
import glob
import scipy.io
import numpy as np

def get_tomo_MFR(data_directory, subsampling = False):
	"""
	Collects the reconstructions performed with MFR 
	algorithm (implemented in Matlab) and saves the needed information
	for the training of the NN in tomo_COMPASS.hdf in the same directory. 
	Only reconstructions with |chi2-1|<0.05 are kept.
	Also stores tomo_GEOM.npz which contains the info about the cameras used

	Inputs:
		data_directory - path directory where reconstructions are saved
		subsampling    - (int) ex. 4 means store at every 4th reconstruction (default is no subsampling)
	"""

	# output *.hdf file name, will be stored in the same directory as the *.mat files
	output_fname = data_directory + 'tomo_COMPASS.hdf'
	new_file = h5py.File(output_fname, 'w')

	n_reconstr = 0

	for filename in glob.glob(data_directory + '*.mat'):

		pulse = filename[filename.find('shot')+len('shot'):filename.find('_reg')]

		print filename

		f = scipy.io.loadmat(filename)

		#------------------------------------------------------------------ 
		# COMPASS Matlab File dictionary keys meaning
		#
		# G_final	- reconstructions (#time_steps, #pixels)
		# tvec		- time vector 
		# CHI2 		- chi^2 values (~1)
		# Y 		- input signal from detectors (#time_steps, #detectors)
		# Yrfit 	- virtual signals from detectors after reconstruction
		# dets 		- list of detectors used
		# dY 		- expected error of detectors
		#------------------------------------------------------------------

		print f.keys()

		t = f['tvec'][:][0]
		tomo =  np.swapaxes(np.swapaxes(f['G_final'][:],0,2),1,2)
		SXR = f['Y'][:]
		eSXR = f['dY'][:]
		CHI2 = f['CHI2'][:,0]
		SXRfit = f['Yrfit'][:]

		print pulse, t.shape, t.dtype, tomo.shape, tomo.dtype, SXR.shape, SXR.dtype, CHI2.shape,\
		eSXR.shape, eSXR.dtype

		index = abs(CHI2-1) < 0.05

		tomo = tomo[index,:,:]
		SXR = SXR[index,:]
		eSXR = eSXR[index,:]
		SXRfit = SXRfit[index,:]
		t = t[index]
		CHI2 = CHI2[index]

		if subsampling :
			
			assert isinstance(subsampling, int)

			index = [i%subsampling==0 for i in range(len(t))]
			tomo = tomo[index,:,:]
			SXR = SXR[index,:]
			eSXR = eSXR[index,:]
			SXRfit = SXRfit[index,:]
			t = t[index]
			CHI2 = CHI2[index]

		n_reconstr += len(t)

		g = new_file.create_group(pulse)
		g.create_dataset('t', data=t)
		g.create_dataset('SXR', data=SXR)
		g.create_dataset('eSXR', data=eSXR)
		g.create_dataset('tomo', data=tomo)
		g.create_dataset('SXRfit', data=SXRfit)
		g.create_dataset('CHI2', data=CHI2)

		print pulse, t.shape, t.dtype, tomo.shape, tomo.dtype, SXR.shape, SXR.dtype, eSXR.shape, eSXR.dtype

	print '# reconstructions :', n_reconstr

	# save detectors and last pulse used. Later it will be needed to know the geometry
	# so the chi2 value can be correctly calculated 
	# -1 due to conversion from matlab to python indeces
	SXRA = np.squeeze(f['dets'][0][0]) - 1
	SXRB = np.squeeze(f['dets'][0][1]) - 1
	SXRF = np.squeeze(f['dets'][0][2]) - 1
	print 'SXRA :', SXRA
	print 'SXRB :', SXRB
	print 'SXRF :', SXRF

	np.savez(data_directory + 'tomo_GEOM.npz', SXRA = SXRA, SXRB = SXRB, SXRF = SXRF, last_pulse = pulse)

def get_pulse_COMPASS(data_directory, pulse, flatten = True):
	"""
	Get Bolometer measures and reconstructions for specific pulse
	Inputs: 
		fname - path to file
		pulse - ID of pulse 
		flatten - If true g is returned flattened
	Outputs:
		t - time of reconstructions [s]
		f_data - measurement from SXR cameras (includes non active) [kW/m^-2]
		g_data - reconstruction [kW/m^-3]
	"""
	fname = data_directory + 'tomo_COMPASS.hdf'

	# check if directory exists
	if not os.path.exists(data_directory):
		sys.exit('Non-existing directory')
	else:
		# check if already exists an *.hdf file, if not create it
		if not os.path.exists():
			print 'Creating *.hdf file ', data_directory + 'tomo_COMPASS.hdf'
			get_tomo_MFR(data_directory, subsampling = subsampling)

	print 'Reading:', fname
	f = h5py.File(fname, 'r')

	f_data = []
	g_data = []

	group = f[pulse]
	t = group['t'][:]
	SXR = group['SXR'][:]/1e3
	tomo = group['tomo'][:]/1e3

	for i in range(len(t)):
		f_data.append(SXR[i])
		if flatten:
			g_data.append(np.asarray(tomo[i,:,:]).flatten())
		else:
			g_data.append(np.asarray(tomo[i,:,:]))

	f.close()

	t = np.asarray(t, dtype = np.float32)
	f_data = np.asarray(f_data, dtype = np.float32)
	g_data = np.asarray(g_data, dtype = np.float32)

	return t, f_data, g_data

def get_tomo_COMPASS(data_directory, flatten = True, subsampling = False):
	"""
	Get Bolometer measures and reconstructions + errors
	Inputs: 
		fname - path to file
		flatten - If true g is returned flattened
	Outputs:
		f_data - measurement from SXR cameras (includes non active) [kW/m^-2]
		g_data - reconstruction [kW/m^-3]
		ef_data - expected error from SXR cameras
		fv_data - virtual bolometers measures [kW/m^-2]
		t - time corresponding to time
	"""

	fname = data_directory + 'tomo_COMPASS.hdf'

	# check if directory exists
	if not os.path.exists(data_directory):
		sys.exit('Non-existing directory')
	else:
		# check if already exists an *.hdf file, if not create it
		if not os.path.exists(fname):
			print 'Creating *.hdf file ', data_directory + 'tomo_COMPASS.hdf'
			get_tomo_MFR(data_directory, subsampling = subsampling)

	print 'Reading:', fname
	f = h5py.File(fname, 'r')

	f_data = []
	g_data = []
	ef_data = []
	fv_data = []
	t_data = []
	chi2_data = []
	pulse_data = []

	for pulse in f:

		group = f[pulse]
		t = group['t'][:]
		SXR = group['SXR'][:]/1e3
		tomo = group['tomo'][:]/1e3
		eSXR = group['eSXR'][:]/1e3
		SXRfit = group['SXRfit'][:]/1e3
		CHI2 = group['CHI2'][:]

		for i in range(len(t)):
			f_data.append(SXR[i])
			ef_data.append(eSXR[i])
			fv_data.append(SXRfit[i])
			chi2_data.append(CHI2[i])
			if flatten:
				g_data.append(np.asarray(tomo[i,:,:]).flatten())
			else:
				g_data.append(np.asarray(tomo[i,:,:]))

			pulse_data.append(pulse)
			t_data.append(t[i])
		
	f.close()

	f_data = np.asarray(f_data, dtype = np.float32)
	g_data = np.asarray(g_data, dtype = np.float32)
	ef_data = np.asarray(ef_data, dtype = np.float32)
	fv_data = np.asarray(fv_data, dtype = np.float32)
	chi2_data = np.asarray(chi2_data, dtype = np.float32)
	pulse_data = np.asarray(pulse_data, dtype = np.float32)
	t_data = np.asarray(t_data, dtype = np.float32)

	return f_data, g_data, ef_data, fv_data,t_data,chi2_data,pulse_data
