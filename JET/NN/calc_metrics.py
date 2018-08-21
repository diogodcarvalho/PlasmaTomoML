
import csv
import numpy as np 

import sys
sys.path.insert(0, '../bib/')
import bib_data
import bib_utils 
import bib_metrics 

# ----------------------------------------------------------------------
# Load Data

fname = '../data/tomo_JET.hdf'
f,g,_,_ = bib_data.get_tomo_JET(fname, faulty = True,  flatten = False, clip_tomo = True)

print 'g:', g.shape, g.dtype
print 'f:', f.shape, f.dtype

# -------------------------------------------------------------------------
# Load sets indices and keep only the validation one
# Choose the directory in wich this information was stored

save_path = './Results/'
indeces = np.load(save_path + './i_divided.npz')

g_test = g[indeces['i_test']]
f_test = f[indeces['i_test']]

print 'g_test:', g_test.shape, g_test.dtype
print 'f_test:', f_test.shape, f_test.dtype

#------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Image quality metrics

ssim,mse,psnr, nrmse = bib_metrics.compare_all_metrics(g_test,g_test_nn, mean = True)

# ------------------------------------------------------------------
# Total Power Emited Error

e_power = (np.sum(g_test,axis = (1,2)) - np.sum(g_test_nn, axis = (1,2)))/np.sum(g_test,axis = (1,2))

#------------------------------------------------------------------
# Mean values

print 'Average Values ------------------------------'
print 'ssim: %.4f +- %.4f' % (np.mean(ssim), np.std(ssim))
print 'psnr: %.2f +- %.2f' % (np.mean(psnr), np.std(psnr))
print 'nrmse: %.2f +- %.2f' % (np.mean(nrmse), np.std(nrmse))
print 'e_power: %.2f +- %.2f' % (np.mean(e_power), np.std(e_power)) 