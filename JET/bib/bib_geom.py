
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from shapely.geometry import MultiLineString, LineString

# Number of lines of sight and pixels
N_LOS = 56
N_ROWS = 196
N_COLS = 115

# Working channels for each camera (THIS INFORMATION WAS TRUE AT THE MOMENT OF WRITTING)
KB5_BROKEN = [15,22] + range(50,56) 
KB5_OFF = range(23,31)
KB5 = list(set(range(0,N_LOS)) - set(KB5_BROKEN) - set(KB5_OFF))

# Detector error (percentage)
KB5_SIGMA = 0.1

# boarders of pixelated image (not pixel center!)
R_MIN = 1.71 - 0.01
R_MAX = 3.99 + 0.01
Z_MIN = -1.77 - 0.01
Z_MAX = +2.13 + 0.01

# pixel spacement 
dr = (R_MAX-R_MIN)/(N_COLS)
dz = (Z_MAX-Z_MIN)/(N_ROWS)

def transform(r, z):
	"""
	Converts vessel coordinates to pixel coordinates
	Inputs:
		r,z - coordinates [m] of the point 
	Outputs:
		(i,j) - coordinates [#pixel] of the point
	"""
	assert z <= Z_MAX and z>= Z_MIN
	assert r <= R_MAX and r>= R_MIN

	i = int((z-Z_MIN)/(Z_MAX-Z_MIN)*(N_ROWS))
	j = int((r-R_MIN)/(R_MAX-R_MIN)*(N_COLS))
	return (i,j)

def get_vessel_JET():
	""" 
	Returns array with Vessel boarder coordinates
	Outputs:
		r - R coordinates
		z - Z coordinates
	"""

	fname = '../geom/geom.txt'
	print 'Reading:', fname
	f = open(fname,'r')

	r = []
	z = []

	for line in f:
		parts = line.split()
		if len(parts) == 2:
			r.append(float(parts[0]))
			z.append(float(parts[1]))

	f.close()
	return r,z

def get_vessel_mask(plot = False):
	""" 
	Returns Vessel Mask
	Inputs:
		plot - if we want to plot mask
	Outputs:
		mask - vessel mask (N_ROWS,N_COLS) if 0 - vessel not there , 1 - otherwise
	"""

	R, Z = get_vessel_JET()

	vessel = [[R[i], Z[i], R[i+1], Z[i+1]] for i in range(len(R)-1)]

	r_space = np.linspace(R_MIN, R_MAX, num = N_COLS+1)
	z_space = np.linspace(Z_MIN, Z_MAX, num = N_ROWS+1)

	grid = []

	for r in r_space:
		grid.append([(r, Z_MIN), (r, Z_MAX)])

	for z in z_space:
		grid.append([(R_MIN, z), (R_MAX, z)])

	grid = MultiLineString(grid)

	mask = np.zeros((N_ROWS, N_COLS))
	for (r_start, z_start, r_end, z_end) in vessel:
		line = LineString([(r_start, z_start), (r_end, z_end)])
		for (k, segment) in enumerate(line.difference(grid)):
			rr, zz = segment.xy
			r_mean = np.mean(rr)
			z_mean = np.mean(zz)
			(i,j) = transform(r_mean, z_mean)
			mask[i,j] = 1.

	mask = np.array(mask)
	print 'mask:', mask.shape, mask.dtype

	if plot:
		plt.figure()
		plt.plot(R, Z, 'r')
		plt.imshow(mask, vmin=0, vmax=np.max(mask), origin = 'lower', extent = [R_MIN, R_MAX, Z_MIN, Z_MAX])
		plt.xlabel('R (m)')
		plt.ylabel('Z (m)')
		plt.show()

	return mask

def get_chamber_mask(plot = False):
	"""
	Returns Chamber Mask (all points inside chamber)
	Inputs:
		plot - if we want to plot image
	Outputs:
		chamber_mask - (N_ROWS,N_COLS) 0 - outside chamber, 1 - inside
	"""
	vessel_mask = get_vessel_mask(plot = False)

	borders = np.asarray(np.flip(vessel_mask,axis=0).nonzero())
	
	chamber_mask = np.zeros(vessel_mask.shape)

	i = 0

	while i < borders.shape[1]:
		n_times = np.asarray(np.nonzero(borders[0] == borders[0,i])).shape[1]
		if borders[0,i] < 189:
			chamber_mask[borders[0,i],borders[1,i]:borders[1,i+n_times-1]+1] = np.ones(borders[1,i+n_times-1]+1-borders[1,i])
		else:
			#divertor region hard-coded
			chamber_mask[189,35:44] = np.ones(44-35)
			chamber_mask[189,48:60] = np.ones(60-48)
			chamber_mask[190,35:43] = np.ones(43-35)
			chamber_mask[190,51:60] = np.ones(60-51)
			chamber_mask[191,32:42] = np.ones(42-32)
			chamber_mask[191,54:63] = np.ones(63-54)
			chamber_mask[192,31:42] = np.ones(42-31)
			chamber_mask[192,55:64] = np.ones(64-55)
			chamber_mask[193,30:36] = np.ones(36-30)
			chamber_mask[193,59:65] = np.ones(65-59)
			chamber_mask[194,61:65] = np.ones(65-61)
			break
			
		i = i+n_times

	chamber_mask = np.flip(chamber_mask, axis = 0)

	if plot:
		R,Z = get_vessel_JET()
		plt.figure()
		plt.imshow(chamber_mask+vessel_mask,origin = 'lower', extent = [R_MIN, R_MAX, Z_MIN, Z_MAX])
		plt.plot(R,Z,'r')
		plt.xlabel('R (m)')
		plt.ylabel('Z (m)')
		plt.show()

	return chamber_mask

def get_los_JET(inside = True):
	"""
	Returns lines of sight ending and starting points
	Inputs:
		inside - if True returns only points inside vessel
	Outputs:
		ri,rz,zi,zf - LOS coordinates (N_los,)
	"""

	ri = []
	rf = []
	zi = []
	zf = []

	fname = '../geom/kb5_los.txt'
	print 'Reading:', fname
	f = open(fname, 'r')

	for line in f:
		parts = line.split()
		if len(parts) == 22:
			if inside:
				ri.append(float(parts[10]))
				zi.append(float(parts[11]))
			else:
				ri.append(float(parts[7]))
				zi.append(float(parts[8]))

			rf.append(float(parts[13]))
			zf.append(float(parts[14]))

	f.close()

	return ri,rf,zi,zf	

def get_los_mask(plot = False, los = 'Full'):
	""" 
	Returns los mask considering no width in los
	Inputs:
		plot - if we want to plot mask
		los - List with los to be calculated (by default 'Full' assumes all of them)
	Outputs:
		mask - vessel mask (N_ROWS,N_COLS) value corresponds to length of the los in each pixel
	"""

	Ri, Rf, Zi, Zf = get_los_JET()	

	if los != 'Full':
		Ri = [Ri[i] for i in los] 
		Rf = [Rf[i] for i in los]
		Zi = [Zi[i] for i in los]
		Zf = [Zf[i] for i in los]

	r_space = np.linspace(R_MIN, R_MAX, num = N_COLS+1)
	z_space = np.linspace(Z_MIN, Z_MAX, num = N_ROWS+1)

	grid = []

	for r in r_space:
		grid.append([(r, Z_MIN), (r, Z_MAX)])

	for z in z_space:
		grid.append([(R_MIN, z), (R_MAX, z)])

	grid = MultiLineString(grid)

	mask = np.zeros((N_LOS,N_ROWS, N_COLS))

	for (los,(r_start, z_start, r_end, z_end)) in enumerate(zip(Ri,Zi,Rf,Zf)):

		line = LineString([(r_start, z_start), (r_end, z_end)])

		for (k, segment) in enumerate(line.difference(grid)):
			rr, zz = segment.xy
			r_mean = np.mean(rr)
			z_mean = np.mean(zz)
			try:
				(i,j) = transform(r_mean, z_mean)
				mask[los,i,j] = segment.length
			except :
				pass

	mask = np.array(mask, dtype = np.float32)
	print 'mask:', mask.shape, mask.dtype

	if plot:
		R,Z = get_vessel_JET()
		plt.figure()
		for (r_start, z_start, r_end, z_end) in zip(Ri,Zi,Rf,Zf):
			plt.plot((r_start,r_end), (z_start,z_end), 'r',lw = .7)

		plt.imshow(np.sum(mask, axis = 0), vmin=0, vmax=np.max(mask), origin = 'lower', extent = [R_MIN, R_MAX, Z_MIN, Z_MAX])
		plt.plot(R,Z)
		plt.xlabel('R (m)')
		plt.ylabel('Z (m)')
		plt.colorbar()
		plt.show()

	return mask

def get_virtual_bolometers(tomo):
	"""
	From existent tomography estimate correspondent 
	bolometer measurement 
	Inputs: 
		tomo - tomographic reconstructions (N_reconstruction, N_ROWS, N_COLS)
	Outputs:
		virtual - virtual bolometer measurement [kW/m^-2] (N_reconstruction, N_LOS)
	"""

	los_mask = get_los_mask()

	assert tomo.shape[1:] == (N_ROWS,N_COLS)	

	tomo = np.expand_dims(tomo, axis=1)

	virtual = tomo*los_mask

	virtual = np.sum(virtual, axis = (-2,-1))
	
	return virtual	

def get_centroid(tomo):
	"""
	Calculates centroid of plasma emissivity 
	Inputs: 
		tomo - tomographic reconstructions (N_reconstruction, N_ROWS, N_COLS) or (N_ROWS, N_COLS)
	Outputs:
		R_centroind, Z_centroid - position of centroid (N_reconstructions,)		
	"""

	R = np.arange(R_MIN+dr/2., R_MAX, dr)
	Z = np.arange(Z_MIN+dr/2., Z_MAX, dz)

	R_centroid = np.sum(tomo, axis = -2)*R
	Z_centroid = np.sum(tomo, axis = -1)*Z 

	R_centroid = np.sum(R_centroid, axis = -1)
	Z_centroid = np.sum(Z_centroid, axis = -1)

	R_centroid /= np.sum(tomo, axis = (-2,-1))
	Z_centroid /= np.sum(tomo, axis = (-2,-1))

	return R_centroid, Z_centroid

def get_chi2(detector, tomo, error_detector):
	"""
	Calculates chi_2 of reconstructions (removes faulty detectors) 
	Inputs: 
		detector - kb5 measures (N_reconstruction, N_LOS) or (N_LOS)
		tomo - tomographic reconstructions (N_reconstruction, N_ROWS, N_COLS) or (N_ROWS, N_COLS)
		error_detector - error associated to detector each detector (N_reconstruction, N_LOS) or (N_LOS)
	Outputs:
		chi2 - chi2 values (N_reconstructions,)		
	"""

	tomo = np.clip(tomo, a_min = 0, a_max = None)

	virtual = get_virtual_bolometers(tomo)

	if detector.ndim == 1:
		detector = detector[np.newaxis,:]
		tomo = tomo[np.newaxis,:]
		error_detector = error_detector[np.newaxis,:]
		detector_virtual = detector_virtual[np.newaxis,:]

	chi2 = virtual-detector

	chi2 = np.divide(chi2, error_detector)

	chi2 = chi2[:,KB5]
	
	chi2 = chi2**2
	
	chi2 = np.sum(chi2, axis = 1)

	chi2 /= np.sum(KB5)

	return chi2
