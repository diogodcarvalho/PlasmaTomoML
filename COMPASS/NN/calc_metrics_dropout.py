
import csv
import numpy as np 

import sys
sys.path.insert(0, '../bib/')
import bib_geom 
import bib_utils
import bib_metrics 

# -------------------------------------------------------------------------
# Initialize geometry

save_path = './Results/'
geom = bib_geom.Geometry(save_path + 'tomo_GEOM.npz')

# -------------------------------------------------------------------------
print '\nLoad data'

tomo_COMPASS = np.load(save_path + 'tomo_COMPASS.npz')
f = tomo_COMPASS['f']
g = tomo_COMPASS['g']
ef = tomo_COMPASS['ef']

print 'g:', g.shape, g.dtype
print 'f:', f.shape, f.dtype
print 'ef:', ef.shape, ef.dtype

# -------------------------------------------------------------------------
print '\nKeep test set'

f_test = f[tomo_COMPASS['i_test']]
g_test = g[tomo_COMPASS['i_test']]
ef_test = ef[tomo_COMPASS['i_test']]

print 'f_test:', f_test.shape
print 'g_test:', g_test.shape
print 'ef_test:', ef_test.shape

# -------------------------------------------------------------------------
print '\nInitialize NN'

import nn_model

# recover number of filters the NN was tested with
with open(save_path + 'model_options.log') as inputfile:
 	for row in csv.reader(inputfile):
 		print row[0]
 		if 'filters' in row[0]:
 			filters = int(row[0][9:])

# build model
model = nn_model.build_model(filters)

# load tested parameters
model_parameters = save_path + 'model_parameters.hdf'
model.load_weights(model_parameters)

# -------------------------------------------------------------------------
# Calculate quality metrics for different combinations of 
# shutdown detectors

print '\nCalculate Metrics'

from copy import deepcopy
from itertools import combinations

# number of shutdown detectors at each time
n_shutdown = 1

# list to store all metric values
variables_i = [[] for i in range(7)] 

# generate combinations of detectors working detectors to shut-down 
shutdown_detectors_combinations = combinations(geom.SXR,n_shutdown)

for shutdown_detectors in shutdown_detectors_combinations:

	sys.stdout.write("\r" + 'camera shutdown: ' + str(shutdown_detectors))
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

print '\nAverage Values'
print 'ssim: %.4f +- %.4f' % (np.mean(ssim), np.std(ssim))
print 'psnr: %.2f +- %.2f' % (np.mean(psnr), np.std(psnr))
print 'nrmse: %.2f +- %.2f' % (np.mean(nrmse), np.std(nrmse))
print 'e_power: %.2f +- %.2f' % (np.mean(e_power), np.std(e_power)) 
print 'e_R: %.2f +- %.2f' % (np.mean(e_R), np.std(e_R))
print 'e_Z: %.2f +- %.2f' % (np.mean(e_Z), np.std(e_Z))
print 'chi2_nn: %.2f +- %.2f' % (np.mean(chi2_nn), np.std(chi2_nn))