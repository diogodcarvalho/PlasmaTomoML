import h5py
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib 

import sys
sys.path.insert(0, '../bib/')
import bib_geom 
import bib_data 

# ----------------------------------------------------------
# Import bolometer measures and reconstructions from a
# choosen pulse. These should already be stored in a file
# that was created from accessing the JET database using
# the script PlasmaTomoML/JET/data/get_bolo.py

pulse = '92213' 

fname = '../data/test_data.hdf'
f,t = bib_data.get_bolo_JET(fname, pulse, faulty = False)

f = f.transpose()

print 'f :', f.shape, f.dtype
print 't :', t.shape, t.dtype

# ----------------------------------------------------------
# Import matrix and perform new reconstructions 
# Choose the directory in wich it was stored
# output files of this script will also be stored there

save_path = './Results/'
M = np.load(save_path + 'M.npy')

g = np.dot(M,f).transpose()
print 'g :', g.shape, g.dtype

# ----------------------------------------------------------
# Import lines of sight and vessel

ri,rf,zi,zf = bib_geom.get_los_JET(inside = False)

r,z = bib_geom.get_vessel_JET()

# ----------------------------------------------------------
# Initialize figure with zeros
# adapt the dynamic range by changing vmin and vmax as you wish

matplotlib.rcParams.update({'font.size': 8})

fig = plt.figure()
ax = plt.axes()
im = ax.imshow(np.zeros((bib_geom.N_ROWS,bib_geom.N_COLS)), vmin = 0,vmax = 1.,
	cmap = 'inferno', origin = 'lower', extent = [bib_geom.R_MIN, bib_geom.R_MAX,bib_geom.Z_MIN,bib_geom.Z_MAX])

# Set plot_los = False if you don't want the LOS to appear
# in the reconstructions
plot_los = True
if plot_los:
	lines = []
	for i,(r1,r2,z1,z2) in enumerate(zip(ri,rf,zi,zf)):
		line, = plt.plot((r1,r2),(z1,z2),'gold')
		lines.append(line)
	
	ttl = ax.text(.25, 1.0, '', transform = ax.transAxes, color='white', fontsize = 8)

else :
	ttl = ax.text(.35, 1.05, '', transform = ax.transAxes, color='white', fontsize = 8)
	

plt.plot(1,-2.,'black')
plt.plot(r,z,'white',linewidth=0.7)


# ----------------------------------------------------------
#  Edit the colorbar

if plot_los:
	cbaxes = fig.add_axes([.8, 0.17, 0.02, 0.63])
else:
	cbaxes = fig.add_axes([.8, 0.19, 0.02, 0.63])

cb = plt.colorbar(im, cax = cbaxes)
cb.ax.set_title('MW/m$^{-3}$', color='white',fontsize = 8)
cb.ax.tick_params(labelsize=8)
cb.ax.yaxis.set_tick_params(color='white')
cb.outline.set_edgecolor('white')
plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')


# ----------------------------------------------------------
#  Edit the axes

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')	
ax.spines['left'].set_linewidth(.5)
ax.spines['bottom'].set_linewidth(.5)
ax.set_facecolor('black')

ax.set_xlabel('R (m)', color = 'grey')
ax.set_ylabel('Z (m)', color = 'grey')

if plot_los:
	ax.xaxis.set_label_coords(0.45,-0.04)
	ax.yaxis.set_label_coords(-.05,.5)
else :
	ax.xaxis.set_label_coords(0.6,-0.04)
	ax.yaxis.set_label_coords(-.05,.55)

ax.xaxis.set_ticks([bib_geom.R_MIN,bib_geom.R_MAX])
ax.yaxis.set_ticks([bib_geom.Z_MIN,bib_geom.Z_MAX])

ax.tick_params(axis='x', colors= 'grey')
ax.tick_params(axis='y', colors= 'grey')

ax.spines['left'].set_bounds(bib_geom.Z_MIN,bib_geom.Z_MAX)
ax.spines['bottom'].set_bounds(bib_geom.R_MIN,bib_geom.R_MAX)


# ----------------------------------------------------------
#  Functions to generate animation

def transform_f_lw(f):
	"""
	Adapt linewidth os LOS depending on the measured value
	"""
	lw = f/2.
	# detectors were faulty at the time of writting
	# and so the line width was set to zero
	# if you want to keep them just comment the two line below
	lw[15] = 0
	lw[22] = 0
	lw = np.clip(lw,a_min = 0, a_max = 1)
	return lw

def init():
	ttl.set_text('')
	im.set_data(np.zeros((N_rows,N_cols)))
	return im,ttl

def updatefig(frame):
	im.set_data(g[frame].reshape((bib_geom.N_ROWS,bib_geom.N_COLS)))
	ttl.set_text('JET pulse %s t=%.2fs' % (pulse, t[frame]))
	lws = transform_f_lw(f[:,frame])
	if plot_los:
		for lw,line in zip(lws,lines):
			line.set_lw(lw)
	print frame, t[frame]
	return im,ttl

# Choose time interval to plot
tmin = t[0]
tmax = t[-1]
frames = [i for i,ti in enumerate(t) if ti<=tmax and ti>=tmin]
print 'frames : i - ', frames[0], ' f - ', frames[-1]

ani = animation.FuncAnimation(fig, updatefig, frames = frames, interval=100, repeat = False,blit=True)

# Choose file where to save animation
fname = save_path + pulse + '.mp4'
ani.save(fname, fps=15, extra_args=['-vcodec', 'libx264'], savefig_kwargs={'facecolor':'black'})

# Choose file where to save the reconstructions
fname = save_path + pulse 
np.savez(save_path + pulse, t = t[frames], f = np.transpose(f)[frames], g = np.transpose(g)[frames])

