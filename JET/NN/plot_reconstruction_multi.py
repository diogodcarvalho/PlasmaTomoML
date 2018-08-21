
import csv
import numpy as np 
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../bib/')
import bib_geom
import bib_data
import bib_utils

# ----------------------------------------------------------
# Import bolometer measures and reconstructions from a
# choosen pulse. These should already be stored in a file
# that was created from accessing the JET database using
# the script PlasmaTomoML/JET/data/get_bolo.py

pulse = '92213' 

fname = '../data/test_data.hdf'
f,t = bib_data.get_bolo_JET(fname, pulse, faulty = True, clip_tomo = True)

print 'f :', f.shape, f.dtype
print 't :', t.shape, t.dtype

tmin = 49.62
tmax = 54.02
frames = [((tmin-tx)<1e-4 or tx>tmin) and ((tx-tmax)<1e-4 or tx<tmax) for tx in t]
t = t[frames]
f = f[frames]

dt = 0.1
frames = [i%int(dt*100) == 0 for i in range(len(t))]
t = t[frames]
f = f[frames]

print 'f :', f.shape, f.dtype
print 't :', t.shape, t.dtype

#------------------------------------------------------------------
# Initialize NN for given trained weights
# Choose the directory in wich it was stored
# output files of this script will also be stored there

import nn_model

save_path = './Results/'

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
g_nn = model.predict(f)
# resize to original resolution
g_nn = bib_utils.resize_NN_image(g_nn, training = False)

print 'g_nn:', g_nn.shape, g_nn.dtype

# -------------------------------------------------------------------------
# Plot grid of reconstructions

font = {'weight' : 'normal',
        'size'   : 3}

plt.rc('font', **font)

nx = 5
ny = 9

from matplotlib import rcParams
rcParams['axes.titlepad'] = 2 

fig, ax = plt.subplots(nx, ny)
for i in range(nx):
    for j in range(ny):

        print 'Pulse %s t=%.4fs' % (pulse, t[i*ny+j])

        ax[i,j].imshow(g_nn[i*ny+j].reshape(bib_geom.N_ROWS,bib_geom.N_COLS), vmin=0, vmax=1.5,
         origin='lower',cmap = 'inferno', extent = [bib_geom.R_MIN, bib_geom.R_MAX, bib_geom.Z_MIN,bib_geom.Z_MAX])
        title = 't=%.2fs' % (t[i*ny+j])
        ax[i,j].set_title(title)
        ax[i,j].axis('off')

# this one works for 5x9
plt.subplots_adjust(left = 0.05, right = 0.55 , bottom=0.3, top=0.9, hspace=0.2)

# this one works for 3x9
#plt.subplots_adjust(left = 0, right = .6 , bottom=0.42, top=0.9, hspace=0, wspace = 0.1)

plt.savefig(save_path + '/JET_' + pulse + '_' + str(tmin) + '_' + str(tmax) + '.png', dpi = 300, bbox_inches='tight')

