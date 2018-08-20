
import numpy as np 
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../bib/')
import bib_metrics
import bib_data
import bib_geom

# -------------------------------------------------------------------------
# Load data 
# File selected needs to be the same one in which the matrix M was fitted
# since we will be using the validation set previously defined

fname = '../data/train_data.hdf'

f,g,_,_ = bib_data.get_tomo_JET(fname, faulty = True,  flatten = False)

print 'g:', g.shape, g.dtype
print 'f:', f.shape, f.dtype

# -------------------------------------------------------------------------
# Load sets indices and keep only the validation one
# Choose the directory in wich this information was stored

save_path = './Results/'
indeces = np.load( save_path + './i_divided.npz')

g_valid = g[indeces['i_valid']]
f_valid = f[indeces['i_valid']]

print 'g_valid:', g_valid.shape, g_valid.dtype
print 'f_valid:', f_valid.shape, f_valid.dtype

# -------------------------------------------------------------------------
# Import matrix and perform new reconstructions (only in validation set)
# Choose the directory in wich it was stored
# output files of this script will also be stored there

save_path = './Results/'
M = np.load(save_path + 'M.npy')

g_m = np.dot(M,f_valid.transpose()).transpose()
g_m = g_m.reshape((g_m.shape[0],bib_geom.N_ROWS,bib_geom.N_COLS))
print 'g_m :', g_m.shape, g_m.dtype

# Can uncomment if you decide to clip mininimum value of images to zero
# g_valid = np.clip(g_valid, a_min = 0, a_max = None)
# g_m = np.clip(g_m, a_min = 0, a_max = None)

#------------------------------------------------------------------

ssim,mse,psnr, nrmse = bib_metrics.compare_all_metrics(g_valid,g_m)
e_power = (np.sum(g_valid,axis = (1,2)) - np.sum(g_m, axis = (1,2)))/np.sum(g_valid,axis = (1,2))

print 'Average Values ------------------------------'
print 'ssim: %.4f +- %.4f' % (np.mean(ssim), np.std(ssim))
print 'psnr: %.2f +- %.2f' % (np.mean(psnr), np.std(psnr))
print 'nrmse: %.2f +- %.2f' % (np.mean(nrmse), np.std(nrmse))
print 'e_power: %.2f +- %.2f' % (np.mean(e_power), np.std(e_power)) 
