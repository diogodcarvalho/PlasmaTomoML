
import numpy as np
np.random.seed(0)

import sys
sys.path.insert(0, '../bib/')
import bib_utils 

# -------------------------------------------------------------------------
print '\nLoad data'

save_path = './Results_virtual/'
tomo_COMPASS = np.load(save_path + 'tomo_COMPASS.npz')
f = tomo_COMPASS['f']
g = tomo_COMPASS['g']

# need to reshape image to match NN dimensions
g = bib_utils.resize_NN_image(g, training = True)

print 'g:', g.shape, g.dtype
print 'f:', f.shape, f.dtype

# -------------------------------------------------------------------------
print '\nDefine training/validation sets'

f_train = f[tomo_COMPASS['i_train']]
g_train = g[tomo_COMPASS['i_train']]
f_valid = f[tomo_COMPASS['i_valid']]
g_valid = g[tomo_COMPASS['i_valid']]

print 'f_train:', f_train.shape
print 'g_train:', g_train.shape
print 'f_valid:', f_valid.shape
print 'g_valid:', g_valid.shape

# -------------------------------------------------------------------------
print '\nInitialize NN'

import nn_model 
import nn_callback 
from keras.optimizers import *

# default values used, change as you see fit
# loss function is mean absolute error
loss = 'mae'
# number of filter in each convolution layer (also changes dense layer size)
filters = 15
# learning rate
lr = 1e-4
# number of epochs
epochs = int(1e5)
# dropout rate at inputs
dropout_rate = 0.
# training set is divided by keras in batches with this size
batch_size = 435
# no prints from keras
verbose = 0

# save options used into a file
logname = save_path + 'model_options'
nn_callback.write('loss: %s\nfilters: %i\nepochs: %i\nlr: %e\ndropout_rate: %.2f\nbatch_size: %i\n'
           % (loss,filters,epochs,lr,dropout_rate,batch_size), logname,reset = True)

train_batches = float(len(f_train))/float(batch_size)
valid_batches = float(len(f_valid))/float(batch_size)

print 'batch_size:', batch_size, '(train: %.3f, valid: %.3f)' % (train_batches, valid_batches)

# Use Adam optimizer
opt = Adam(lr=lr)

# Load and compile NN model
model = nn_model.build_model(filters = filters, dropout_rate = dropout_rate)
model.compile(loss=loss, optimizer=opt)

print '\nStart Training'

# start training
# 2 files will be created
# train.log - where loss function values are stored
# model_parameters.hdf - where best parameter values for the NN are stored
# the training will only stop after the maximum number of epochs is reached.
# you can also press ctrl+c to stop it earlier and the output files will present
# the best values until that point
model.fit(x=f_train,
          y=g_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=verbose,
          callbacks=[nn_callback.MyCallback(save_path)],
          validation_data=(f_valid,g_valid))