
import os
import time
import numpy as np
import matplotlib.pyplot as plt 

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

# -------------------------------------------------------------------------
# Load data 

fname = '../data/train_data.hdf'

# if one wants to use the faulty detectors values as is change to fauly = True
# this will reduce the convergence rate of the algorithm leading to a bigger number of
# epoch needed, in this scenario increase later to the following values
# when initializing gradient descent : epochs = 1e6, momentum = 0.999
f,g,_,_ = bib_data.get_tomo_JET(fname, faulty = False,  flatten = True)

g = np.transpose(g)*1e3
f = np.transpose(f)*1e3

print 'g:', g.shape, g.dtype
print 'f:', f.shape, f.dtype

# ------------------------------------------------------------------------
# Divide into training and validation set 
# (no test set needed since it won't overfit)
# if one wants the validation set can also be disregarded by setting ratio=[1.,0.]

i_train, i_valid, _ = bib_utils.divide_data(g.shape[1],ratio = [.9,.1],test_set = False,random = False)
g_valid = g[:,i_valid]
f_valid = f[:,i_valid]
g = g[:,i_train]
f = f[:,i_train]

print 'g_train:', g.shape, g.dtype
print 'f_train:', f.shape, f.dtype
print 'g_valid:', g_valid.shape, g_valid.dtype
print 'f_valid:', f_valid.shape, f_valid.dtype

np.savez(save_path + 'i_divided', i_train = i_train, i_valid = i_valid)

# ------------------------------------------------------------------------
# Initialize matrix with zeros

M = np.zeros((g.shape[0],f.shape[0]), dtype=np.float32)
print 'M:', M.shape, M.dtype

# -------------------------------------------------------------------------
# Initialize theano variables

import theano
import theano.tensor as T
from theano.printing import pydotprint

g = theano.shared(g, 'g')
f = theano.shared(f, 'f')

M = theano.shared(M, 'M')
loss = T.mean(T.abs_(T.dot(M,f)-g))
grad = T.grad(loss, M)

# -------------------------------------------------------------------------
# Momentum gradient descent implementation

# the values of the learning rate, momentum and epochs might need to be
# adjusted for different data-sets for a better convergence

learning_rate = np.float32(.01)
momentum = np.float32(0.9)

print 'learning_rate:', learning_rate
print 'momentum:', momentum

updates = []

m = theano.shared(M.get_value() * np.float32(0.))
v = momentum * m - learning_rate * grad

updates.append((m, v))
updates.append((M, M + momentum * v - learning_rate * grad))

# -------------------------------------------------------------------------
# Run gradient decent for a given number of epochs

train = theano.function(inputs=[],
                        outputs=[loss],
                        updates=updates)

pydotprint(train, outfile= save_path + 'train.png', compact=False)

epochs = int(1e5)

# saves the loss functions values to a *.log file
fname = save_path + 'train.log'
print 'Writing:', fname
f = open(fname, 'w')

# training can be interrupted at any time by pressing ^c
# all logs until that point and curret matrix obtained will be saved
try:
    for epoch in range(epochs):
        if epoch == 0:
            s = '%-10s %10s %10s' % ('time', 'epoch', 'loss (kW/m^3)')
            print s
            f.write(s)
            f.write('\n')
            f.flush()
        outputs = train()
        loss_value = outputs[0]
        t = time.strftime('%H:%M:%S')
        s = '%-10s %10d %10.6f' % (t, epoch, loss_value)
        if epoch%100 == 0:
            print s
        f.write(s)
        f.write('\n')
        f.flush()

except KeyboardInterrupt:
    print 'Training interrupted.'

f.close()

# -------------------------------------------------------------------------
# Save matrix obtained 

M = M.get_value()

print 'M:', M.shape, M.dtype

fname = save_path + 'M.npy'
print 'Writing:', fname
np.save(fname, M)

# -------------------------------------------------------------------------
# Plot loss function during training

log_loss = np.loadtxt(save_path + 'train.log',skiprows = 1,usecols = (1,2))

i = log_loss[:,0]
loss_train = log_loss[:,1]

plt.figure()
plt.plot(i/10**3,loss_train,label = 'training')
plt.xlabel('epoch ($10^3$)')
plt.ylabel('$\mathcal{L}$ (kW m$^{-3}$)')
plt.grid(True)
plt.savefig(save_path + 'loss_log.png',dpi = 300, bbox_inches='tight')

    

