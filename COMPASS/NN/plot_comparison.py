
import os
import csv
import numpy as np 

import sys
sys.path.insert(0, '../bib/')
import bib_geom
import bib_data
import bib_utils

# -------------------------------------------------------------------------
print '\nLoad data'

save_path = './Results/'
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

f_test = f[tomo_COMPASS['i_test']]
g_test = g[tomo_COMPASS['i_test']]
t_test = t[tomo_COMPASS['i_test']]
pulse_test = pulse[tomo_COMPASS['i_test']]

print 'g_test:', g_test.shape, g_test.dtype
print 'f_test:', f_test.shape, f_test.dtype
print 't_test:', t_test.shape, t_test.dtype
print 'pulse_test:', pulse_test.shape, pulse_test.dtype

if not os.path.exists(save_path + 'COMPARE/'):
    print 'Creating directory ', save_path + 'COMPARE/'
    os.makedirs(save_path + 'COMPARE/')

# -------------------------------------------------------------------------
# Initialize NN for given trained weights

import nn_model

# recover number of filters the NN was trained with
with open(save_path + 'model_options.log') as inputfile:
    for row in csv.reader(inputfile):
        print row[0]
        if 'filters' in row[0]:
            filters = int(row[0][9:])

# build model
model = nn_model.build_model(filters)

# load trained parameters
model_parameters = save_path + 'model_parameters.hdf'
model.load_weights(model_parameters)

# calculate new reconstructions with the NN
g_test_nn = model.predict(f_test)
# resize to original resolution
g_test_nn = bib_utils.resize_NN_image(g_test_nn, training = False)

print 'g_test:', g_test.shape, g_test.dtype
print 'g_test_nn:', g_test_nn.shape, g_test_nn.dtype

# -------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

cmapX2 = LinearSegmentedColormap.from_list('name', ['darkblue', 'darkcyan','turquoise',
                                            'white', 'tomato', 'red', 'firebrick'])

font = {'weight' : 'normal',
        'size'   : 6}

plt.rc('font', **font)

# 'All' plots every tomogram from testation set
pulses_to_plot = 'All'
# or you can specify a list of pulses to plot (example below)
# pulses_to_plot = [10167,10169]

if pulses_to_plot != 'All':
    i_to_plot = [i for i,pulse in enumerate(pulse_test) if pulse in pulses_to_plot]
else:
    i_to_plot = range(pulse_test.shape[0])

for i in i_to_plot:

    title = 'Pulse %i t=%.2fs' % (pulse_test[i], t_test[i])
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
    # im = ax2.imshow(g_test_nni/10**3, vmin=0, vmax=.5, cmap = 'inferno', extent = [bib_geom.R_MIN, bib_geom.R_MAX, bib_geom.Z_MIN, bib_geom.Z_MAX])
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
    ax1.imshow(g_test[i], vmin=0, vmax=5., origin = 'lower',  cmap = 'inferno', extent = [bib_geom.R_MIN, bib_geom.R_MAX, bib_geom.Z_MIN, bib_geom.Z_MAX])
    ax1.set_xlabel('R (m)')
    ax1.set_ylabel('Z (m)')
    ax1.set_title('Original', fontdict = font)
    fig.add_subplot(ax1) 
    ax2 = plt.Subplot(fig,inner1[1])
    im2 = ax2.imshow(g_test_nn[i], vmin=0, vmax=5., origin = 'lower', cmap = 'inferno', extent = [bib_geom.R_MIN, bib_geom.R_MAX, bib_geom.Z_MIN, bib_geom.Z_MAX])
    ax2.set_xlabel('R (m)')
    ax2.set_title('NN', fontdict = font)
    cax2 = fig.add_axes([0.607,0.585,0.01, 0.33])
    clb2 = fig.colorbar(im2, cax=cax2,format='%.1f')
    clb2.ax.set_title('kW m$^{-3}$',fontdict = font)
    fig.add_subplot(ax2)

    ax3 = plt.Subplot(fig,outer[1])
    im3 = ax3.imshow(g_test[i]-g_test_nn[i], vmin=-1.5, vmax=1.5, origin = 'lower', cmap = cmapX2, extent = [bib_geom.R_MIN, bib_geom.R_MAX, bib_geom.Z_MIN, bib_geom.Z_MAX])
    ax3.set_xlabel('R (m)')
    ax3.set_title('Difference', fontdict = font)
    cax3 = fig.add_axes([0.92,0.585,0.01, 0.33])
    clb3 = fig.colorbar(im3, cax=cax3,format='%.1f')
    clb3.ax.set_title('kW m$^{-3}$',fontdict = font)
    fig.add_subplot(ax3)

    fig.subplots_adjust(bottom = 0.05, right = 0.9, top = 1.45,hspace=0,wspace = 0.3)

    plt.savefig(save_path + 'COMPARE/COMPASS_' + str(pulse_test[i]) + '_' + str(t_test[i]) + '.png',dpi=300,bbox_inches='tight')
    plt.close()