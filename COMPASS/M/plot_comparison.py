
import os
import numpy as np 

import sys
sys.path.insert(0, '../bib/')
import bib_geom

# -------------------------------------------------------------------------
print '\nLoad data'

save_path = './Results_virtual/'
tomo_COMPASS = np.load(save_path + 'tomo_COMPASS.npz')
f = tomo_COMPASS['f']
g = tomo_COMPASS['g']
t = tomo_COMPASS['t']
pulse = tomo_COMPASS['pulse']

print 'g:', g.shape, g.dtype
print 'f:', f.shape, f.dtype
print 't:', t.shape, t.dtype
print 'pulse:', pulse.shape, pulse.dtype 

# -------------------------------------------------------------------------
print '\nKeep test set'

f_valid = f[tomo_COMPASS['i_valid']]
g_valid = g[tomo_COMPASS['i_valid']]
t_valid = t[tomo_COMPASS['i_valid']]
pulse_valid = pulse[tomo_COMPASS['i_valid']]

print 'g_valid:', g_valid.shape, g_valid.dtype
print 'f_valid:', f_valid.shape, f_valid.dtype
print 't_valid:', t_valid.shape, t_valid.dtype
print 'pulse_valid:', pulse_valid.shape, pulse_valid.dtype

if not os.path.exists(save_path + 'COMPARE/'):
    print 'Creating directory ', save_path + 'COMPARE/'
    os.makedirs(save_path + 'COMPARE/')

# -------------------------------------------------------------------------
print '\nCalculate Reconstructions'

M = np.load(save_path + 'M.npy')

g_m = np.dot(M,f_valid.transpose()).transpose()
g_m = g_m.reshape((g_m.shape[0],bib_geom.N_ROWS,bib_geom.N_COLS))

print 'g_m :', g_m.shape, g_m.dtype

# Uncomment if you decide to clip mininimum value of images to zero
# g_valid = np.clip(g_valid, a_min = 0, a_max = None)
# g_m = np.clip(g_m, a_min = 0, a_max = None)

# -------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

cmapX2 = LinearSegmentedColormap.from_list('name', ['darkblue', 'darkcyan','turquoise',
                                            'white', 'tomato', 'red', 'firebrick'])

font = {'weight' : 'normal',
        'size'   : 6}

plt.rc('font', **font)

# 'All' plots every tomogram from validation set
pulses_to_plot = 'All'
# or you can specify a list of pulses to plot (example below)
# pulses_to_plot = [10167,10169]

if pulses_to_plot != 'All':
    i_to_plot = [i for i,pulse in enumerate(pulse_valid) if pulse in pulses_to_plot]
else:
    i_to_plot = range(pulse_valid.shape[0])

for i in i_to_plot:

    title = 'Pulse %i t=%.2fs' % (pulse_valid[i], t_valid[i])
    print title

    fig = plt.figure()
    fig.suptitle(title)

    # Version without difference between images
    # ax1 = fig.add_subplot(121)  
    # ax1.imshow(g_i/10**3, vmin=0, vmax=.5, origin = 'lower',  cmap = 'inferno', extent = [bib_geom.R_MIN, bib_geom.R_MAX, bib_geom.Z_MIN, bib_geom.Z_MAX])
    # ax1.set_xlabel('R (m)')
    # ax1.set_ylabel('Z (m)')
    # ax1.set_title('Original')
    # ax2 = fig.add_subplot(122)
    # im = ax2.imshow(g_mi/10**3, vmin=0, vmax=.5, cmap = 'inferno', extent = [bib_geom.R_MIN, bib_geom.R_MAX, bib_geom.Z_MIN, bib_geom.Z_MAX])
    # ax2.set_xlabel('R (m)')
    # ax2.set_title('Inverse Matrix')
    # fig.subplots_adjust(bottom = 0.1, right = 0.9, top = 0.9)
    # cax = fig.add_axes([0.95,0.1,0.02, 0.8])
    # clb = fig.colorbar(im, cax=cax,format='%.1f')
    # clb.ax.set_title('MW m$^{-3}$')   

    # Version with difference between images
    outer = gridspec.GridSpec(1,2, width_ratios = [2,.83], wspace=0.38)
    
    inner1 = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec = outer[0], wspace = 0.4)
    ax1 = plt.Subplot(fig, inner1[0])  
    ax1.imshow(g_valid[i], vmin=0, vmax=5., origin = 'lower',  cmap = 'inferno', extent = [bib_geom.R_MIN, bib_geom.R_MAX, bib_geom.Z_MIN, bib_geom.Z_MAX])
    ax1.set_xlabel('R (m)')
    ax1.set_ylabel('Z (m)')
    ax1.set_title('Original', fontdict = font)
    fig.add_subplot(ax1) 
    ax2 = plt.Subplot(fig,inner1[1])
    im2 = ax2.imshow(g_m[i], vmin=0, vmax=5., origin = 'lower', cmap = 'inferno', extent = [bib_geom.R_MIN, bib_geom.R_MAX, bib_geom.Z_MIN, bib_geom.Z_MAX])
    ax2.set_xlabel('R (m)')
    ax2.set_title('Inverse Matrix', fontdict = font)
    cax2 = fig.add_axes([0.607,0.585,0.01, 0.33])
    clb2 = fig.colorbar(im2, cax=cax2,format='%.1f')
    clb2.ax.set_title('kW m$^{-3}$',fontdict = font)
    fig.add_subplot(ax2)

    ax3 = plt.Subplot(fig,outer[1])
    im3 = ax3.imshow(g_valid[i]-g_m[i], vmin=-1.5, vmax=1.5, origin = 'lower', cmap = cmapX2, extent = [bib_geom.R_MIN, bib_geom.R_MAX, bib_geom.Z_MIN, bib_geom.Z_MAX])
    ax3.set_xlabel('R (m)')
    ax3.set_title('Difference', fontdict = font)
    cax3 = fig.add_axes([0.92,0.585,0.01, 0.33])
    clb3 = fig.colorbar(im3, cax=cax3,format='%.1f')
    clb3.ax.set_title('kW m$^{-3}$',fontdict = font)
    fig.add_subplot(ax3)

    fig.subplots_adjust(bottom = 0.05, right = 0.9, top = 1.45,hspace=0,wspace = 0.3)

    plt.savefig(save_path + 'COMPARE/COMPASS_' + str(pulse_valid[i]) + '_' + str(t_valid[i]) + '.png',dpi=300,bbox_inches='tight')
    plt.close()