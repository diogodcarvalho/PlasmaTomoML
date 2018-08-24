
import csv
import numpy as np 

import sys
sys.path.insert(0, '../bib/')
import bib_data
import bib_geom
import bib_utils
import bib_metrics

# -------------------------------------------------------------------------
# Load data from tomograms obtained with MFR

data_directory = '../data/Reconstructions/'
f, g, ef,_,t,chi2,pulse = bib_data.get_tomo_COMPASS(data_directory,  flatten = False)

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
pulse_test = pulse[indeces['i_valid']]
t_test = t[indeces['i_valid']]
chi2_test = chi2[indeces['i_valid']]

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

# calculate new reconstructions with the NN
g_test_nn = model.predict(f_test)
# resize to original resolution
g_test_nn = bib_utils.resize_NN_image(g_test_nn, training = False)

print 'g_test:', g_test.shape, g_test.dtype
print 'g_test_nn:', g_test_nn.shape, g_test_nn.dtype

# ------------------------------------------------------------------
# Image metrics

ssim,mse,psnr, nrmse = bib_metrics.compare_all_metrics(g_test,g_test_nn, mean = False)

# ------------------------------------------------------------------
# Pixelwise Metrics

rmse_pixel,amre_pixel,mre_pixel = bib_metrics.compare_all_metrics_pixel(g_test,g_test_nn, mean = True)

bib_metrics.plot_metric_pixel(rmse_pixel, 'RMSE', save_path,vmin = 0, vmax = 0.4,clb_legend = 'kW m$^{-3}$')
bib_metrics.plot_metric_pixel(mre_pixel, 'MRE', save_path,vmin = -.15, vmax = .15)
bib_metrics.plot_metric_pixel(amre_pixel, 'AMRE', save_path,vmin = 0, vmax = 0.30)

# ------------------------------------------------------------------
# Total Power Emited Error

e_power = (np.sum(g_test,axis = (1,2)) - np.sum(g_test_nn, axis = (1,2)))/np.sum(g_test,axis = (1,2))

#------------------------------------------------------------------
# Check Plasma Center

R_centroid, Z_centroid  = bib_geom.get_centroid(g_test)
R_nn_centroid, Z_nn_centroid = bib_geom.get_centroid(g_test_nn)

R_error = np.abs(R_centroid-R_nn_centroid)*1e3
Z_error = np.abs(Z_centroid-Z_nn_centroid)*1e3

#------------------------------------------------------------------
# Chi2 Values

geom = bib_geom.Geometry(data_directory + 'tomo_GEOM.npz')
chi2_m = geom.get_chi2(f_test,g_test_nn,ef_test)

#------------------------------------------------------------------
# Mean values

print '\nAverage Values ------------------------------'
print 'ssim: %.4f +- %.4f' % (np.mean(ssim), np.std(ssim))
print 'psnr: %.2f +- %.2f' % (np.mean(psnr), np.std(psnr))
print 'nrmse: %.2f +- %.2f' % (np.mean(nrmse), np.std(nrmse))
print 'e_power: %.2f +- %.2f' % (np.mean(e_power), np.std(e_power)) 
print 'R_error: %.2f +- %.2f' % (np.mean(R_error), np.std(R_error)) 
print 'Z_error: %.2f +- %.2f' % (np.mean(Z_error), np.std(Z_error))
print 'chi2_m: %.2f +- %.2f' % (np.mean(chi2_m), np.std(chi2_m))


#------------------------------------------------------------------
# Save results to file *.csv for further analysis

print '\nCreating :', save_path + 'metrics.csv'

with open(save_path + 'metrics.csv', 'w') as csvfile:
	fieldnames = ['pulse', 't', 'ssim', 'psnr', 'nrmse', 'e_power', 'R_error', 'Z_error', 'chi2_m']
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

	writer.writeheader()
	for i in range(pulse_test.shape[0]):
		writer.writerow({'pulse': pulse_test[i], 't': t_test[i] , 'ssim': ssim[i],
		 'psnr': psnr[i], 'nrmse': nrmse[i], 'e_power': e_power[i], 
		 'R_error': R_error[i], 'Z_error': Z_error[i], 'chi2_m': chi2_m[i]})

#------------------------------------------------------------------
# Debuggin sometimes needed

# sometimes COMPASS chi2 from MFR does not match 
# the one calculated with bib_geom.py at all
# it seems this happens when there is saturation of detectors
# which generates poor quality reconstructions 
# these reconstructions should not taken into account
chi2_2 = geom.get_chi2(f_test,g_test,ef_test)

from collections import Counter

if np.mean(np.abs(chi2_test-chi2_2))>0.1:
	print '\n-----------------ATTENTION---------------------'
	print 'Original chi2 from MFR Matlab'
	print 'chi2: %.5f +- %.5f' % (np.mean(chi2_test), np.std(chi2_test))  
	print 'Value calculated for same reconstructions but with bib_geom.py code'
	print 'chi2_2: %.5f +- %.5f' % (np.mean(chi2_2), np.std(chi2_2))
	print("""\nThe values of chi2_2 and chi2 should be similar
If not it might indicate there are problems with
MFR chi2 calculation. Check the following pulses and consider 
removing the problematic reconstructions since they most 
probably belong to events where detectors saturated.""")

	# one way to identify such shots
	# also checking the metrics.csv file might help
	problematic_events = np.abs(chi2_2-1)> 0.1
	problematic_pulses = pulse_test[problematic_events]
	problems_dict = Counter(problematic_pulses)
	print '\nProblematic pulses'
	for pulse in problems_dict.keys():
		print '%i \t - %i \t events' % (int(pulse), problems_dict[pulse])


