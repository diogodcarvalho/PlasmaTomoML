
import csv
import numpy as np 

import sys
sys.path.insert(0, '../bib/')
import bib_data
import bib_utils
import bib_geom 
import bib_metrics 

# ----------------------------------------------------------------------
# Load Data

fname = '../data/train_data_NN.hdf'
f,g,_,_ = bib_data.get_tomo_JET(fname, faulty = True,  flatten = False, clip_tomo = True)

print 'g:', g.shape, g.dtype
print 'f:', f.shape, f.dtype

# -------------------------------------------------------------------------
# Load sets indices and keep only the validation one
# Choose the directory in wich this information was stored

save_path = './Results/'
indeces = np.load( save_path + './i_divided.npz')

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

#------------------------------------------------------------------
# Calculate quality metrics for different combinations of 
# shutdown detectors

from copy import deepcopy
from itertools import combinations

# number of shutdown detectors at each time
n_shutdown = 2

# list to store all metric values
variables_i = [[] for i in range(4)] 

# generate combinations of detectors to shut-down 
shutdown_detectors_combinations = combinations([i for i in range(bib_geom.N_LOS)],n_shutdown)

for shutdown_detectors in shutdown_detectors_combinations:

	sys.stdout.write("\r" + 'cameras: ' + str(shutdown_detectors))
	sys.stdout.flush()

	f_dropout = deepcopy(f_test)

	# shut-down detectors
	for detector in shutdown_detectors:
		f_dropout[:, detector] = np.zeros(f_dropout.shape[0], dtype = np.float32)

	# perform new reconstructions
	g_test_nn = model.predict(f_dropout)
	g_test_nn = bib_utils.resize_NN_image(g_test_nn,training = False)

	# Image metrics
	ssim,mse,psnr, nrmse = bib_metrics.compare_all_metrics(g_test,g_test_nn, mean = True)

	# Total Power Emited Error
	e_power = np.abs((np.sum(g_test,axis = (1,2)) - np.sum(g_test_nn, axis = (1,2)))/np.sum(g_test,axis = (1,2)))

	variables = [ssim, psnr, nrmse, e_power]
	for variable,variable_i in zip(variables,variables_i):
		variable_i.append(variable)

ssim = np.asarray(variables_i[0])
psnr = np.asarray(variables_i[1]) 
nrmse = np.asarray(variables_i[2]) 
power = np.asarray(variables_i[3]) 

#------------------------------------------------------------------
# Mean values

print '\nAverage Values ------------------------------'
print 'ssim: %.4f +- %.4f' % (np.mean(ssim), np.std(ssim))
print 'psnr: %.2f +- %.2f' % (np.mean(psnr), np.std(psnr))
print 'nrmse: %.2f +- %.2f' % (np.mean(nrmse), np.std(nrmse))
print 'e_power: %.2f +- %.2f' % (np.mean(e_power), np.std(e_power)) 