
import os
import numpy as np
np.random.seed(0)

import sys
sys.path.insert(0, '../bib/')
import bib_data
import bib_utils 

# -------------------------------------------------------------------------
# Directory to which all results will be saved, changed as you which 

save_path = './Results/'
if not os.path.exists(save_path):
        print 'Creating directory ', save_path
        os.makedirs(save_path)

# ----------------------------------------------------------------------
# Load Data

fname = '../data/tomo_JET.hdf'
f,g,_,_ = bib_data.get_tomo_JET(fname, faulty = True,  flatten = False, clip_tomo = True)

# need to reshape image to match NN dimensions
g = bib_utils.resize_NN_image(g, training = True)

print 'g:', g.shape, g.dtype
print 'f:', f.shape, f.dtype

# ------------------------------------------------------------------------
# Divide into training, validation and test set 

i_train, i_valid, i_test = bib_utils.divide_data(g.shape[0],ratio = [.8,.1,.1],test_set = True,random = False)

f_valid = f[i_valid]
g_valid = g[i_valid]
f_train = f[i_train]
g_train = g[i_train]

print 'f_train:', f_train.shape
print 'g_train:', g_train.shape
print 'f_valid:', f_valid.shape
print 'g_valid:', g_valid.shape

np.savez(save_path + 'i_divided', i_train = i_train, i_valid = i_valid, i_test = i_test)

# ----------------------------------------------------------------------
# Train the NN

import nn_model 
import nn_callback 
from keras.optimizers import *

# default values used, change as you see fit
# loss function is mean absolute error
loss = 'mae'
# number of filter in each convolution layer
filters = 20
# learning rate
lr = 1e-4
# number of epochs
epochs = int(1e5)
# dropout rate at inputs
dropout_rate = 0.
# self explanatory
batch_size = 398
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
model = nn_model.build_model(filters,dropout_rate)
model.compile(loss=loss, optimizer=opt)

# start training
# 2 files will be created
# train.log - where loss function values are stored
# model_parameters.hdf - where best parameter values for the NN are stored
model.fit(x=f_train,
          y=g_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=verbose,
          callbacks=[nn_callback.MyCallback(save_path)],
          validation_data=(f_valid,g_valid))
