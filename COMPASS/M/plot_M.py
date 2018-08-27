
import os
import numpy as np 
import matplotlib.pyplot as plt 

import sys
sys.path.insert(0, '../bib/')
import bib_geom 

# -------------------------------------------------------------------------

save_path = './Results/'
# all *.png files will be stored in the following directory
if not os.path.exists(save_path + 'LOS/'):
    print 'Creating directory ', save_path + 'LOS/'
    os.makedirs(save_path + 'LOS/')

# -------------------------------------------------------------------------

M = np.load(save_path + 'M.npy')
print 'M :', M.shape, M.dtype

# -------------------------------------------------------------------------
print '\nLoad vessel coordinates and lines of sight'

r,z = bib_geom.get_vessel_COMPASS()
geom = bib_geom.Geometry(save_path + 'tomo_GEOM.npz')
ri,rf,zi,zf = geom.get_los_COMPASS_2()

# -------------------------------------------------------------------------
print '\nPlot regularization'

font = {'weight' : 'normal',
		'size'   : 11}

plt.rc('font', **font)

# if simple = False colorbar and labels are added
simple = True

for i in range(M.shape[1]):

	print 'LOS ', i, M[:,i].shape, M[:,i].max(), M[:,i].min(), M[:,i].std()

	fig = plt.figure()
	# adjust the dynamic range as you wish
	plt.imshow(M[:,i].reshape(bib_geom.N_ROWS,bib_geom.N_COLS), origin = 'lower', vmin = 0, vmax = 1., cmap = 'inferno',  extent = [bib_geom.R_MIN,bib_geom.R_MAX,bib_geom.Z_MIN,bib_geom.Z_MAX])
	plt.plot(r,z,'g')
	plt.plot((ri[i],rf[i]),(zi[i],zf[i]),'r--')
	plt.xlim((bib_geom.R_MIN,bib_geom.R_MAX))
	plt.ylim((bib_geom.Z_MIN,bib_geom.Z_MAX))
	
	plt.title('LOS' + str(i+1))

	if not(simple):
		plt.xlabel('R (m)')
		plt.ylabel('Z (m)')
		clb = plt.colorbar(format= '%.1f',aspect=30)
		clb.ax.set_title('m$^{-1}$')

	else:
		plt.axis('off')

	# choose either .svg or png
	plt.savefig(save_path + 'LOS/COMPASS_LOS_' + str(i+1) + '.png', dpi = 300, bbox_inches='tight',transparent=True, pad_inches=0)
	plt.close()
	# plt.savefig(save_path + 'LOS/COMPASS_LOS_' + str(i+1) + '.svg', bbox_inches='tight',transparent=True, pad_inches=0)
