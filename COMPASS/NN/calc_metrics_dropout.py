
import csv
import numpy as np 

import sys
sys.path.insert(0, '../bib/')
import bib_data
import bib_utils
import bib_geom 
import bib_metrics 

# -------------------------------------------------------------------------
# Load data from tomograms obtained with MFR

data_directory = '../data/Reconstructions/'
f, g, ef,_,_,_,_ = bib_data.get_tomo_COMPASS(data_directory,  flatten = False)

print 'g:', g.shape, g.dtype
print 'f:', f.shape, f.dtype
print 'ef:', ef.shape, ef.dtype

# -------------------------------------------------------------------------
# Load sets indices and keep only the validation one
# Choose the directory in wich this information was stored

save_path = './Results/'
indeces = np.load( save_path + './i_divided.npz')

g_test = g[indeces['i_valid']]
f_test = f[indeces['i_valid']]
ef_test = ef[indeces['i_valid']]

print 'g_test:', g_test.shape, g_test.dtype
print 'f_test:', f_test.shape, f_test.dtype
print 'ef_test:', ef_test.shape, ef_test.dtype

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
# Initialize geometry

geom = bib_geom.Geometry(data_directory + 'tomo_GEOM.npz')

#------------------------------------------------------------------
# Calculate quality metrics for different combinations of 
# shutdown detectors

from copy import deepcopy
from itertools import combinations

# number of shutdown detectors at each time
n_shutdown = 2

# list to store all metric values
variables_i = [[] for i in range(7)] 

# generate combinations of detectors working detectors to shut-down 
shutdown_detectors_combinations = combinations(geom.SXR,n_shutdown)

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
	ssim,mse,psnr, nrmse = bib_metrics.compare_all_metrics(g_test,g_test_nn)

	# Total Power Emited Error
	e_power = np.abs((np.sum(g_test,axis = (1,2)) - np.sum(g_test_nn, axis = (1,2)))/np.sum(g_test,axis = (1,2)))

	# Check Plasma Center
	R_centroid, Z_centroid  = bib_geom.get_centroid(g_test)
	R_nn_centroid, Z_nn_centroid = bib_geom.get_centroid(g_test_nn)

	e_R = np.abs(R_centroid-R_nn_centroid)*1e3
	e_Z = np.abs(Z_centroid-Z_nn_centroid)*1e3

	# Chi2 Values
	chi2_nn = geom.get_chi2(f_test,g_test_nn,ef_test)

	variables = [ssim, psnr, nrmse, e_power, e_R, e_Z, chi2_nn]
	for variable,variable_i in zip(variables,variables_i):
		variable_i.append(variable)

ssim = np.asarray(variables_i[0])
psnr = np.asarray(variables_i[1]) 
nrmse = np.asarray(variables_i[2]) 
e_power = np.asarray(variables_i[3]) 
e_R = np.asarray(variables_i[4])
e_Z = np.asarray(variables_i[5])
chi2_nn = np.asarray(variables_i[6])

#------------------------------------------------------------------
# Mean values

print '\nAverage Values ------------------------------'
print 'ssim: %.4f +- %.4f' % (np.mean(ssim), np.std(ssim))
print 'psnr: %.2f +- %.2f' % (np.mean(psnr), np.std(psnr))
print 'nrmse: %.2f +- %.2f' % (np.mean(nrmse), np.std(nrmse))
print 'e_power: %.2f +- %.2f' % (np.mean(e_power), np.std(e_power)) 
print 'e_R: %.2f +- %.2f' % (np.mean(e_R), np.std(e_R))
print 'e_Z: %.2f +- %.2f' % (np.mean(e_Z), np.std(e_Z))
print 'chi2_nn: %.2f +- %.2f' % (np.mean(chi2_nn), np.std(chi2_nn))