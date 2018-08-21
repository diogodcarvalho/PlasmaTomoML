
import os
import numpy as np 

import sys
sys.path.insert(0, '../bib/')
import bib_metrics 
import bib_geom
import bib_data

# -------------------------------------------------------------------------
# Load data 
# File selected needs to be the same one in which the matrix M was fitted
# since we will be using the validation set previously defined

fname = '../data/tomo_JET.hdf'

f,g,t,pulse = bib_data.get_tomo_JET(fname, faulty = True,  flatten = False)

print 'g:', g.shape, g.dtype
print 'f:', f.shape, f.dtype
print 't:', t.shape, t.dtype
print 'pulse:', pulse.shape, pulse.dtype 

# -------------------------------------------------------------------------
# Load sets indices and keep only the validation one
# Choose the directory in wich this information was stored
# output files of this script will also be stored there

save_path = './Results/'
indeces = np.load( save_path + './i_divided.npz')

g_valid = g[indeces['i_valid']]
f_valid = f[indeces['i_valid']]
t_valid = t[indeces['i_valid']]
pulse_valid = pulse[indeces['i_valid']]

print 'g_valid:', g_valid.shape, g_valid.dtype
print 'f_valid:', f_valid.shape, f_valid.dtype
print 't_valid:', t_valid.shape, t_valid.dtype
print 'pulse_valid:', pulse_valid.shape, pulse_valid.dtype

if not os.path.exists(save_path + 'COMPARE/'):
    print 'Creating directory ', save_path + 'COMPARE/'
    os.makedirs(save_path + 'COMPARE/')

# -------------------------------------------------------------------------
# Import matrix and perform new reconstructions (only in validation set)
# Choose the directory in wich it was stored
# output files of this script will also be stored there

save_path = './Results/'
M = np.load(save_path + 'M.npy')

g_m = np.dot(M,f_valid.transpose()).transpose()
g_m = g_m.reshape((g_m.shape[0],bib_geom.N_ROWS,bib_geom.N_COLS))
print 'g_m :', g_m.shape, g_m.dtype

# -------------------------------------------------------------------------
# Plot original reconstructions vs new reconstructions

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

cmapX2 = LinearSegmentedColormap.from_list('name', ['darkblue', 'darkcyan','turquoise', 'white', 'tomato', 'red', 'firebrick'])

font1 = {'weight' : 'normal',
        'size'   : 5}

font2 = {'weight' : 'normal',
        'size'   : 7}

plt.rc('font', **font1)

SSIM  = []
PSNR  = []
NRMSE = []
POWER = []

for pulse,t,g_mi,g_i in zip(pulse_valid,t_valid,g_m,g_valid):

    title = 'Pulse %i t=%.2fs' % (pulse, t) 
    print title

    ssim,_,psnr, nrmse = bib_metrics.compare_all_metrics(g_i,g_mi)
    e_power = (np.sum(g_i) - np.sum(g_mi))/np.sum(g_i)
    print '%.4f %.2f %.2f %.2f' % (ssim[0],psnr[0],nrmse[0],e_power)

    SSIM.append(ssim)
    PSNR.append(psnr)
    NRMSE.append(nrmse)
    POWER.append(e_power)

    fig = plt.figure()
    fig.suptitle(title, fontsize = 7)

    # Version without difference between images
    # ax1 = fig.add_subplot(121)  
    # ax1.imshow(g_i, vmin=0, vmax=.5, origin = 'lower',  cmap = 'inferno', extent = [bib_geom.R_MIN, bib_geom.R_MAX, bib_geom.Z_MIN, bib_geom.Z_MAX])
    # ax1.set_xlabel('R (m)')
    # ax1.set_ylabel('Z (m)')
    # ax1.set_title('Original')
    # ax2 = fig.add_subplot(122)
    # im = ax2.imshow(g_mi, vmin=0, vmax=.5, origin = 'lower', cmap = 'inferno', extent = [bib_geom.R_MIN, bib_geom.R_MAX, bib_geom.Z_MIN, bib_geom.Z_MAX])
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
    ax1.imshow(g_i, vmin=0, vmax=.5, origin = 'lower',  cmap = 'inferno', extent = [bib_geom.R_MIN, bib_geom.R_MAX, bib_geom.Z_MIN, bib_geom.Z_MAX])
    ax1.set_xlabel('R (m)')
    ax1.set_ylabel('Z (m)')
    ax1.set_title('Original', fontdict = font2)
    fig.add_subplot(ax1) 
    ax2 = plt.Subplot(fig,inner1[1])
    im2 = ax2.imshow(g_mi, vmin=0, vmax=.5, origin = 'lower', cmap = 'inferno', extent = [bib_geom.R_MIN, bib_geom.R_MAX, bib_geom.Z_MIN, bib_geom.Z_MAX])
    ax2.set_xlabel('R (m)')
    ax2.set_title('Inverse Matrix', fontdict = font2)
    cax2 = fig.add_axes([0.605,0.48,0.01, 0.435])
    clb2 = fig.colorbar(im2, cax=cax2,format='%.1f')
    clb2.ax.set_title('MW m$^{-3}$',fontdict = font1)
    fig.add_subplot(ax2)

    ax3 = plt.Subplot(fig,outer[1])
    im3 = ax3.imshow((g_i-g_mi), origin = 'lower', vmin=-.2, vmax=.2, cmap = cmapX2, extent = [bib_geom.R_MIN, bib_geom.R_MAX, bib_geom.Z_MIN, bib_geom.Z_MAX])
    ax3.set_xlabel('R (m)')
    ax3.set_title('Difference', fontdict = font2)
    cax3 = fig.add_axes([0.92,0.48,0.01, 0.435])
    clb3 = fig.colorbar(im3, cax=cax3,format='%.1f', ticks = np.linspace(-.2,.2,5))
    clb3.ax.set_title('MW m$^{-3}$',fontdict = font1)
    fig.add_subplot(ax3)

    fig.subplots_adjust(bottom = 0.05, right = 0.9, top = 1.35,hspace=0,wspace = 0.3)

    plt.savefig(save_path + 'COMPARE/JET_' + str(pulse) + '_' + str(t) + '.png',dpi=300,bbox_inches='tight')
    plt.close()

    
print 'Average Values ------------------------------'
print 'ssim: %.4f +- %.4f' % (np.mean(SSIM), np.std(SSIM))
print 'psnr: %.2f +- %.2f' % (np.mean(PSNR), np.std(PSNR))
print 'nrmse: %.2f +- %.2f' % (np.mean(NRMSE), np.std(NRMSE))
print 'power: %.2f +- %.2f' % (np.mean(POWER), np.std(POWER))  