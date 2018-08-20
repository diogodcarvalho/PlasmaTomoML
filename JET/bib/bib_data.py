
import h5py
import numpy as np

from bib_geom import KB5_BROKEN

def get_pulse_JET(fname, pulse, faulty = True, flatten = True):
	"""
	Get Bolometer measures and reconstructions for a specific pulse
	Inputs: 
		fname - path to file
		pulse - ID of pulse 
		faulty - if true faulty detectors keep their values (set to zero otherwise)
		flatten - If true g is returned flattened
	Outputs:
		f_data - measurement from kb5 [MW/m^-2]
		g_data - reconstruction [MW/m^-3]
		t - time of reconstructions [s]
	"""

	print 'Reading:', fname
	f = h5py.File(fname, 'r')

	f_data = []
	g_data = []

	group = f[pulse]
	t = group['t'][:]
	kb5 = group['kb5'][:]/1e6
	tomo = group['tomo'][:]/1e6

	if not(faulty):
		for i in KB5_BROKEN:
			kb5[:,i] = 0

	for i in range(len(t)):
		f_data.append(kb5[i])
		if flatten:
			g_data.append(np.flip(np.asarray(tomo[i,:,:]),axis = 0).flatten())
		else:
			g_data.append(np.flip(np.asarray(tomo[i,:,:]),axis = 0))

	f.close()

	t = np.asarray(t, dtype = np.float32)
	f_data = np.asarray(f_data, dtype = np.float32)
	g_data = np.asarray(g_data, dtype = np.float32)

	return f_data, g_data, t

def get_tomo_JET(fname, faulty = True,  flatten = True):
	"""
	Get Bolometer measures and reconstructions + errors
	Inputs: 
		fname - path to file
		faulty - if true faulty detectors keep their values (set to zero otherwise)
		flatten - If true g is returned flattened
	Outputs:
		f_data - measurement from KB5 [MW/m^-2]
		g_data - reconstructions [MW/m^-3]
		t_data - time corresponding to the reconstruction [s]
		pulse_data - pulse corresponding to the reconstruction 
	"""

	print 'Reading:', fname
	f = h5py.File(fname, 'r')

	f_data = []
	g_data = []
	t_data = []
	pulse_data = []

	for pulse in f:

		group = f[pulse]
		t = group['t'][:]
		kb5 = group['kb5'][:]
		tomo = group['tomo'][:]


		if not(faulty):
			for i in KB5_BROKEN:
				kb5[:,i] = 0

		for i in range(len(t)):
			f_data.append(kb5[i])
			if flatten:
				g_data.append(np.flip(np.asarray(tomo[i,:,:]),axis = 0).flatten())
			else:
				g_data.append(np.flip(np.asarray(tomo[i,:,:]),axis = 0))

			pulse_data.append(pulse)
			t_data.append(t[i])
		
	f.close()

	f_data = np.asarray(f_data, dtype = np.float32)/1e6
	g_data = np.asarray(g_data, dtype = np.float32)/1e6
	t_data = np.asarray(t_data, dtype = np.float32)
	pulse_data = np.asarray(pulse_data, dtype = np.int32)

	return f_data, g_data,t_data, pulse_data

def get_tomo_JET_v2(fname = '../data/train_data.hdf', faulty = False, flatten = True):
	"""
	Get Bolometer measures and reconstructions
	Inputs: 
		fname - path to file
		flatten - If true g is returned flattened
	Outputs:
		f_data - measurement from SXR cameras (includes non active) [kW/m^-2]
		g_data - reconstruction [kW/m^-3]
	"""

	print 'Reading:', fname
	f = h5py.File(fname, 'r')

	f_data = []
	g_data = []

	for pulse in f:
		group = f[pulse]
		t = group['bolo_t'][:]
		kb5 = group['bolo'][:]/1e3
		tomo = group['tomo'][:]/1e3

		if not(faulty):
			for i in KB5_BROKEN:
				kb5[:,i] = 0

		for i in range(len(t)):
			f_data.append(kb5[i])
			if flatten:
				g_data.append(np.flip(np.asarray(tomo[i,:,:]),axis = 0).flatten())
			else:
				g_data.append(np.flip(np.asarray(tomo[i,:,:]),axis = 0))

	f.close()

	f_data = np.asarray(f_data, dtype = np.float32)
	g_data = np.asarray(g_data, dtype = np.float32)

	return f_data, g_data

def get_bolo_JET(fname, pulse, faulty = True):
	"""
	Get solely Bolometer measures 
	Inputs: 
		fname - path to file
		pulse - the pulse from which we want the bolometer values
		faulty - if False sets the values of faulty detectores to zero
	Outputs:
		f_data - measurement from bolometers (includes non active) [MW/m^-2]
		t - time corresponding to the reconstruction [s]
	"""

	print 'Reading:', fname
	f = h5py.File(fname, 'r')

	f_data = []
	
	group = f[pulse]
	t = group['t'][:]
	kb5 = group['bolo'][:]/1e6

	if not(faulty):
		for i in KB5_BROKEN:
			kb5[:,i] = 0

	for i in range(len(t)):
		f_data.append(kb5[i])

	f.close()

	f_data = np.asarray(f_data, dtype = np.float32)
	t = np.asarray(t, dtype = np.float32)

	return f_data,t
