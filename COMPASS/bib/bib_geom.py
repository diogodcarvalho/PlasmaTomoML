import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

import scipy.io

from shapely.geometry import MultiLineString, LineString

# Number of lines of sight and pixels
N_ROWS = 135
N_COLS = 105

# boarders of pixelated image (not pixel center!)
R_MIN = 0.2
R_MAX = 0.9
Z_MIN = -0.45
Z_MAX = 0.45

# pixel widths
DR = (R_MAX-R_MIN)/(N_COLS)
DZ = (Z_MAX-Z_MIN)/(N_ROWS)

R_TOMO = np.linspace(R_MIN+DR/2., R_MAX-DR/2., num = N_COLS)
Z_TOMO = np.linspace(Z_MIN+DZ/2., Z_MAX-DZ/2., num = N_ROWS)

#--------------------------------------------------------------------
# SXR setup independent

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

def get_vessel_COMPASS(fname = '../geom/border.txt'):
	""" 
	Returns Vessel coordinates
	Outputs:
		r - R coordinates
		z - Z coordinates
	"""

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

	R, Z = get_vessel_COMPASS()

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
		chamber_mask[borders[0,i],borders[1,i]:borders[1,i+n_times-1]+1] = np.ones(borders[1,i+n_times-1]+1-borders[1,i])

		i = i+n_times

	chamber_mask = np.flip(chamber_mask, axis = 0)

	if plot:
		R,Z = get_vessel_COMPASS()
		plt.figure()
		plt.imshow(chamber_mask+vessel_mask,origin = 'lower', extent = [R_MIN, R_MAX, Z_MIN, Z_MAX])
		plt.plot(R,Z,'r')
		plt.show()

	return chamber_mask


def get_centroid(tomo):
	"""
	Calculates centroid of plasma emissivity 
	Inputs: 
		tomo - tomographic reconstructions (N_reconstruction, N_ROWS, N_COLS) or (N_ROWS, N_COLS)
	Outputs:
		R_centroind, Z_centroid - position of centroid (N_reconstructions,)		
	"""

	R = np.arange(R_MIN+DR/2., R_MAX, DR)
	Z = np.arange(Z_MIN+DZ/2., Z_MAX, DZ)

	R_centroid = np.sum(tomo, axis = -2)*R
	Z_centroid = np.sum(tomo, axis = -1)*Z 

	R_centroid = np.sum(R_centroid, axis = -1)
	Z_centroid = np.sum(Z_centroid, axis = -1)

	R_centroid /= np.sum(tomo, axis = (-2,-1))
	Z_centroid /= np.sum(tomo, axis = (-2,-1))

	return R_centroid, Z_centroid


def plot_full_pulse(g, t, pulse, vmin = 0, vmax = 1.5*10**3, output_file = 'anim_pulse'):
	"""
	Creates an animation of a full pulse reconstruction saving it to an .mp4 file
	Inputs:
		g - tomographic reconstructions
		t - time vector associated with g
		pulse - correspondent pulse
		vmin - 
		vmax - 
		output_file - path + name of file where animation (.mp4) should be stored no need for file extension
	"""

	fig = plt.figure()
	ax = plt.axes()
	plt.axis('off')
	im = ax.imshow(np.zeros((N_ROWS,N_COLS)), origin = 'lower', vmin = vmin,vmax = vmax, cmap = 'inferno')
	ttl = ax.text(.1, 1.05, '', transform = ax.transAxes)

	def init():
		ttl.set_text('')
		im.set_data(np.zeros((N_ROWS,N_COLS)))
		return im,ttl

	def updatefig(frame):
		print frame, np.max(g[frame])
		im.set_data(g[frame])
		ttl.set_text('Pulse %s t = %.2f s' % (pulse, t[frame]))
		return im,ttl

	ani = animation.FuncAnimation(fig, updatefig, frames = range(0,len(t)), interval=100, repeat = False,blit=True)
	ani.save(output_file+ '.mp4', fps=15, extra_args=['-vcodec', 'libx264'])


#--------------------------------------------------------------------
# SXR setup Dependent

FIRST_SHOT_GEOM201410 = 8468
LAST_SHOT_GEOM201410 = 13212

FIRST_SHOT_GEOM201701 = 13213
LAST_SHOT_GEOM201701 = 14883

FIRST_SHOT_GEOM201708 = 14884
LAST_SHOT_GEOM201708 = 16217

N_LOS_MAX = 90

class Geometry:
	"""This class handles all the calculations related to line of sight geometry being it
	contribution matrix, virtual detectors, chi2 calculation. 

	Attributes:
		geometry - SXR setup geometry 
		N_LOS- maximum number of lines of sigh for the given geometry (includes non-selected cameras)
		SXRA - working detectors from camera A (0-35)
		SXRB - working detectors from camera B (35-70)
		SXRF - working detectors from camera F (70-90)
		SXR  - full list of working detectors
		SXR_bool - array of booleans with size 90 ((0 -> LOS not used, 1-> LOS used)
		detector_id - auxiliar to define name of geometry files to open ('0'->'A','1'->'B','5'->'F')
		los_mask - line of sight mask for the given geometry aka contribution matrix 
				   (includes non-selected cameras)
	"""
	def __init__(self, geometry_file):
		"""
		Inputs:
			geometry_file - *.npz file with information about lines of sight used 
							and one of the pulses (to locate geometry)
		"""
		geometry_info = np.load(geometry_file)

		last_pulse = int(geometry_info['last_pulse'])
		self.SXRA = geometry_info['SXRA']
		self.SXRB = geometry_info['SXRB']
		self.SXRF = geometry_info['SXRF']

		if last_pulse < FIRST_SHOT_GEOM201410:
			sys.exit('Geometry still neeeds to be implemented')
		elif last_pulse < LAST_SHOT_GEOM201410:
			self.geometry = 201410
			self.N_LOS = 90
			self.detector_id = ['0','1','5']
		elif last_pulse < LAST_SHOT_GEOM201701:
			self.geometry = 201701
			self.N_LOS = 70
			self.detector_id =  ['0','1']
			self.SXRF = np.asarray([])
		elif last_pulse < LAST_SHOT_GEOM201708:
			self.geometry = 201708
			self.N_LOS = 70
			self.detector_id = ['0','1']
			self.SXRF = np.asarray([])
		else:
			sys.exit('Geometry still neeeds to be implemented')

		print '\nInitializing Geometry :', self.geometry
		print 'SXRA :', self.SXRA
		print 'SXRB :', self.SXRB
		print 'SXRF :', self.SXRF
		
		self.SXR = self.SXRA.tolist() + self.SXRB.tolist() + self.SXRF.tolist()
		self.SXR_bool =  [i in self.SXR for i in range(N_LOS_MAX)]
		self.los_mask = self.get_los_mask()

	def get_los_COMPASS(self):
		"""
		Returns lines of sight geometry considering points outside chamber
		Outputs:
			ri,rz,zi,zf - LOS coordinates (N_los,) , includes points out of chamber
		"""

		ri = []
		rf = []
		zi = []
		zf = []

		for i in self.detector_id:
			fnamex = '../geom/' + str(self.geometry) + '/detector_' + i + '_x.txt'
			fnamey = '../geom/' + str(self.geometry) + '/detector_' + i + '_y.txt'
			print 'Reading:', fnamex, fnamey
			fx = open(fnamex,'r')
			fy = open(fnamey,'r')
			for j, (linex,liney) in enumerate(zip(fx,fy)):
				partsx = linex.split()
				partsy = liney.split()
				if len(partsx) == 2 and len(partsy) == 2:
					ri.append(float(partsx[0]))
					rf.append(float(partsx[1]))
					zi.append(float(partsy[0]))
					zf.append(float(partsy[1]))

			fx.close()
			fy.close()

		return ri,rf,zi,zf	

	def get_los_COMPASS_2(self):
		"""
		Returns lines of sight geometry considering only points inside chamber
		Outputs:
			ri,rz,zi,zf - LOS coordinates (N_los,) , only points inside chamber
		"""

		R, Z = get_vessel_COMPASS()
		vessel = [[R[i], Z[i], R[i+1], Z[i+1]] for i in range(len(R)-1)]

		Ri,Rf,Zi,Zf = self.get_los_COMPASS()

		los_r = []
		los_z = []

		for ri,rf,zi,zf in zip(Ri,Rf,Zi,Zf):
			los = LineString([(ri, zi), (rf, zf)])
			for (r_start, z_start, r_end, z_end) in vessel:
				line = LineString([(r_start, z_start), (r_end, z_end)])
				if line.intersects(los):
					point = line.intersection(los)
					los_r.append(point.x)
					los_z.append(point.y)
					
		los = [[los_r[i],los_r[i+1],los_z[i],los_z[i+1]] for i in range(0,2*self.N_LOS,2)]
		
		los = np.asarray(los)

		return los[:,0], los[:,1], los[:,2], los[:,3]

	def get_los_COMPASS_width(self, N = 10, plot = False):
		"""
		Returns lines of sight geometry considering viewing cone
		Inputs:
			N - number of lines considered in the vieweing cone
			plot - True if we want to save figures for each los
		Outputs:
			ri,rz,zi,zf - LOS coordinates (N_los,N) , includes points
						  out of chamber. Each los has N different associated lines
		"""

		Ri, Rf, Zi, Zf = self.get_los_COMPASS()
		angle = []

		for i in self.detector_id:
			fname = '../geom/' + str(self.geometry) + '/detector_' + i + '_det_angles.txt'
			print 'Reading:', fname
			f = open(fname,'r')
			for line in f:
				angle.append(float(line))
			f.close()

		angle = [[angle[j]/float(i) for i in range(1,N+1)] for j in range(self.N_LOS)]
		angle = np.asarray(angle)

		print angle.shape

		Ri = np.array(Ri, ndmin = 2).transpose()
		Rf = np.array(Rf, ndmin = 2).transpose()
		Zi = np.array(Zi, ndmin = 2).transpose()
		Zf = np.array(Zf, ndmin = 2).transpose()

		print Ri.shape, Rf.shape, Zi.shape, Zf.shape
		Theta = np.arctan((Rf-Ri)/(Zf-Zi))
		D_if  = np.sqrt((Ri-Rf)**2+(Zi-Zf)**2)
		print Theta.shape, D_if.shape

		dR = D_if*np.sin(Theta)*np.sin(np.deg2rad(angle/2.))
		dZ = D_if*np.cos(Theta)*np.sin(np.deg2rad(angle/2.))
		print dR.shape, dR.shape

		R1f = Rf - dR
		R2f = Rf + dR
		Z1f = Zf + dZ
		Z2f = Zf - dZ

		print R1f.shape, R2f.shape, Z1f.shape, Z2f.shape

		if plot :
			for j in range(self.N_LOS):
				plt.figure()
				plt.plot((Ri[j],Rf[j]),(Zi[j],Zf[j]),'g')
				for r1f,r2f,z1f,z2f in zip(R1f[j,:],R2f[j,:],Z1f[j,:],Z2f[j,:]):		
					plt.plot((Ri[j],r1f),(Zi[j],z1f),'r')
					plt.plot((Ri[j],r2f),(Zi[j],z2f),'b')	
				
				print j	
				plt.xlim(R_MIN,R_MAX)
				plt.ylim(Z_MIN,Z_MAX)
				plt.axes().set_aspect('equal')
				plt.savefig(str(j)+'.png')

		Rf = np.concatenate((Rf,R1f,R2f),axis = 1)
		Zf = np.concatenate((Zf,Z1f,Z2f),axis = 1)
		
		print Rf.shape, Zf.shape

		return Ri, Rf, Zi, Zf

	def get_los_mask(self, plot = False, los = 'Full', in_vessel = False):
		""" 
		Returns los mask considering no width in los
		Inputs:
			plot - if we want to plot mask
			los - List with los to be calculated (by default 'Full' assumes all of them)
			in_vessel - True - only points inside chamber, False - Full los
		Outputs:
			mask - vessel mask (N_ROWS,N_COLS) value corresponds to length of los in pixel
		"""
		if in_vessel:
			Ri, Rf, Zi, Zf = self.get_los_COMPASS_2()
		else:
			Ri, Rf, Zi, Zf = self.get_los_COMPASS()	

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

		mask = np.zeros((self.N_LOS,N_ROWS, N_COLS))

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
		mask = np.pad(mask,((0,N_LOS_MAX-mask.shape[0]),(0,0),(0,0)),'constant', constant_values= 0)
		print 'mask:', mask.shape, mask.dtype

		if plot:
			R,Z = get_vessel_COMPASS()
			plt.figure()
			for (r_start, z_start, r_end, z_end) in zip(Ri,Zi,Rf,Zf):
				plt.plot((r_start,r_end), (z_start,z_end), 'r')

			plt.imshow(np.sum(mask, axis = 0), vmin=0, vmax=np.max(mask), origin = 'lower', extent = [R_MIN, R_MAX, Z_MIN, Z_MAX])
			plt.plot(R,Z)
			plt.colorbar()
			plt.show()

		return mask

	def get_los_mask_width(self,plot = False, los = 'Full', N = 100):
		""" 
		Returns los mask considering width in los
		Inputs:
			plot - if we want to plot mask
			los - List with los to be calculated (by default 'Full' assumes all of them)
			N - number of virtual chords per los
		Outputs:
			mask - vessel mask (N_ROWS,N_COLS) if 0 - vessel not there , 1 - otherwise
		"""

		Ri, Rf, Zi, Zf = self.get_los_COMPASS_width(N)

		print Ri.shape, Zi.shape, Rf.shape, Zf.shape

		if los != 'Full':
			Ri = np.asarray([Ri[i] for i in los])
			Rf = np.asarray([Rf[i] for i in los])
			Zi = np.asarray([Zi[i,:] for i in los])
			Zf = np.asarray([Zf[i,:] for i in los])

		print Ri.shape, Zi.shape, Rf.shape, Zf.shape	

		r_space = np.linspace(R_MIN, R_MAX, num = N_COLS+1)
		z_space = np.linspace(Z_MIN, Z_MAX, num = N_ROWS+1)

		grid = []

		for r in r_space:
			grid.append([(r, Z_MIN), (r, Z_MAX)])

		for z in z_space:
			grid.append([(R_MIN, z), (R_MAX, z)])

		grid = MultiLineString(grid)

		mask = np.zeros((self.N_LOS,N_ROWS, N_COLS))

		for (l,(r_start, z_start)) in enumerate(zip(Ri,Zi)):
			print 'los :', l
			for (r_end, z_end) in zip(Rf[l,:],Zf[l,:]):
			
				line = LineString([(r_start, z_start), (r_end, z_end)])

				for (k, segment) in enumerate(line.difference(grid)):
					rr, zz = segment.xy
					r_mean = np.mean(rr)
					z_mean = np.mean(zz)
					try:
						(i,j) = transform(r_mean, z_mean)
						mask[l,i,j] += segment.length
					except :
						pass

		mask = np.array(mask, dtype = np.float32)/float(2*N)
		mask = np.pad(mask,((0,N_LOS_MAX-mask.shape[0]),(0,0),(0,0)),'constant', constant_values= 0)
		print 'mask:', mask.shape, mask.dtype

		if plot:	
				
			for (l,(r_start, z_start,r_end,z_end)) in enumerate(zip(Ri,Zi,Rf[:,0],Zf[:,0])):
				plt.figure()
				plt.plot((r_start,r_end), (z_start,z_end), 'r')
				plt.imshow(mask[l,:,:], vmin=0, vmax=np.max(mask[l,:,:]), extent = [R_MIN, R_MAX, Z_MIN, Z_MAX])
				plt.colorbar()
				plt.savefig(str(l) + '.png')

		return mask 


	def get_virtual_cameras(self,tomo, only_working = False, clip_zero = True):
		"""
		From existent tomography estimate correspondent 
		detector measurement 
		Inputs: 
			tomo - tomographic reconstructions (N_reconstruction, N_ROWS, N_COLS)
			only_working - if True, non-working detectors are set to zero
		Outputs:
			virtual - virtual detector measurement [kW/m^-2] (N_reconstruction, self.N_LOS)
		"""

		if clip_zero:
			tomo = np.clip(tomo, a_min = 0, a_max = None)

		# case where only one reconstruction is provided
		if tomo.ndim == 2:
			assert tomo.shape == (N_ROWS,N_COLS)
			tomo = np.expand_dims(np.expand_dims(tomo, axis = 0),axis = 1)

		# case where multiple reconstructions are provided
		elif tomo.ndim == 3:
			assert tomo.shape[1:] == (N_ROWS,N_COLS)	
			tomo = np.expand_dims(tomo, axis=1)

		# arbitrary size choosen so memory does not blow-up
		max_dim = 1000
		if tomo.shape[0] > max_dim:
			
			n_parts = int(tomo.shape[0]/max_dim)
			
			virtual = self.get_virtual_cameras(tomo[0:max_dim,:,:],only_working,clip_zero)

			for i in range(1,n_parts):
				virtual_part = self.get_virtual_cameras(tomo[i*max_dim:(i+1)*max_dim],only_working,clip_zero)
				virtual = np.concatenate((virtual, virtual_part), axis = 0)
				
			virtual_part = self.get_virtual_cameras(tomo[n_parts*max_dim:],only_working,clip_zero)
			virtual = np.concatenate((virtual, virtual_part), axis = 0)

		else: 	
			virtual = tomo*self.los_mask	
			virtual = np.sum(virtual, axis = (-2,-1))

			if only_working:
				virtual = virtual*self.SXR_bool

		return virtual

	def get_chi2(self,detector, tomo, error_detector):

		#tomo = np.clip(tomo, a_min = 0, a_max = None)

		virtual = self.get_virtual_cameras(tomo)

		if detector.ndim == 1:
			detector = detector[np.newaxis,:]
			tomo = tomo[np.newaxis,:]
			error_detector = error_detector[np.newaxis,:]
			virtual = virtual[np.newaxis,:]

		chi2 = virtual-detector

		chi2 = np.divide(chi2, error_detector)

		chi2 = chi2[:,self.SXR_bool]
		
		chi2 = chi2**2
		
		chi2 = np.sum(chi2, axis = 1)

		chi2 /= np.float32(np.sum(self.SXR_bool))

		return chi2

	def plot_COMPASS_SXR(self):
		
		Ri,Rf,Zi,Zf = self.get_los_COMPASS()
		Ri2,Rf2,Zi2,Zf2 = self.get_los_COMPASS_2()
		R,Z = get_vessel_COMPASS()

		plt.figure()
		for sxr,ri,rf,zi,zf,ri2,rf2,zi2,zf2 in zip(self.SXR_bool, Ri,Rf,Zi,Zf,Ri2,Rf2,Zi2,Zf2):
			
			if sxr:
				plt.plot((ri,ri2),(zi,zi2),'r')
				plt.plot((ri2,rf2),(zi2,zf2),'r')
			else:
				plt.plot((ri,ri2),(zi,zi2),'g')
				plt.plot((ri2,rf2),(zi2,zf2),'g')
		
		plt.plot(R, Z, 'b--')
		plt.axes().set_aspect('equal')
		plt.show()
