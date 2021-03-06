
import os
import numpy as np 
import matplotlib.pyplot as plt 

import sys
sys.path.insert(0, '../bib/')
import bib_geom 

# -------------------------------------------------------------------------
# Load M from directory to which all results from fit_M.py were saved 
# All outputs generated by this program will be stored in the same directory

save_path = './Results/'
M = np.load(save_path + 'M.npy')
print 'M :', M.shape, M.dtype

if not os.path.exists(save_path + 'LOS/'):
    print 'Creating directory ', save_path + 'LOS/'
    os.makedirs(save_path + 'LOS/')

# -------------------------------------------------------------------------
# Load vessel coordinates and lines of sight

r,z = bib_geom.get_vessel_JET()
ri,rf,zi,zf = bib_geom.get_los_JET()

# -------------------------------------------------------------------------
# Plot regularization for each line of sight

font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 11}

plt.rc('font', **font)

# if simple = False colorbar and labels are added
simple = True

for i in range(M.shape[1]):

	print i, M[:,i].shape, M[:,i].max(), M[:,i].min(), M[:,i].std()

	fig = plt.figure()
	# adjust the dynamic range as you wish
	plt.imshow(M[:,i].reshape(bib_geom.N_ROWS,bib_geom.N_COLS), origin = 'lower', vmin = 0, vmax = 1., cmap = 'inferno',  extent = [bib_geom.R_MIN,bib_geom.R_MAX,bib_geom.Z_MIN,bib_geom.Z_MAX])
	plt.plot(r,z,'g')
	plt.plot((ri[i],rf[i]),(zi[i],zf[i]),'r--')
	plt.xlim((bib_geom.R_MIN,bib_geom.R_MAX))
	plt.ylim((bib_geom.Z_MIN,bib_geom.Z_MAX))
	
	if i<32:
		plt.title('KB5V LOS' + str(i+1))
	else:
		plt.title('KB5H LOS' + str(i-32+1))

	if not(simple):
		plt.xlabel('R (m)')
		plt.ylabel('Z (m)')
		clb = plt.colorbar(format= '%.1f',aspect=30)
		clb.ax.set_title('m$^{-1}$')

	else:
		plt.axis('off')

	# choose either .svg or png
	plt.savefig(save_path + 'LOS/JET_LOS_' + str(i+1) + '.png', dpi = 300, bbox_inches='tight',transparent=True, pad_inches=0)
	# plt.savefig(save_path + 'LOS/JET_LOS_' + str(i+1) + '.svg', bbox_inches='tight',transparent=True, pad_inches=0)
