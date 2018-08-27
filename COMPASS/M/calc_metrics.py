
import csv
import numpy as np 

import sys
sys.path.insert(0, '../bib/')
import bib_geom
import bib_metrics

# -------------------------------------------------------------------------
# Initialize Geometry 

save_path = './Results/'
geom = bib_geom.Geometry(save_path + 'tomo_GEOM.npz')

# -------------------------------------------------------------------------
print '\nLoad data'

tomo_COMPASS = np.load(save_path + 'tomo_COMPASS.npz')
f = tomo_COMPASS['f']
g = tomo_COMPASS['g']
ef = tomo_COMPASS['ef']
t = tomo_COMPASS['t']
chi2 = tomo_COMPASS['chi2']
pulse = tomo_COMPASS['pulse']

print 'g:', g.shape, g.dtype
print 'f:', f.shape, f.dtype
print 'ef:', ef.shape, ef.dtype

# -------------------------------------------------------------------------
print '\nKeep test set'

f_valid = f[tomo_COMPASS['i_valid']]
g_valid = g[tomo_COMPASS['i_valid']]
ef_valid = ef[tomo_COMPASS['i_valid']]
t_valid = t[tomo_COMPASS['i_valid']]
chi2_valid = chi2[tomo_COMPASS['i_valid']]
pulse_valid = pulse[tomo_COMPASS['i_valid']]

g_valid = g_valid.reshape((g_valid.shape[0],bib_geom.N_ROWS,bib_geom.N_COLS))

print 'f_valid:', f_valid.shape
print 'g_valid:', g_valid.shape
print 'ef_valid:', ef_valid.shape

# -------------------------------------------------------------------------
print '\nCalculate Reconstructions'

M = np.load(save_path + 'M.npy')

g_m = np.dot(M,f_valid.transpose()).transpose()
g_m = g_m.reshape((g_m.shape[0],bib_geom.N_ROWS,bib_geom.N_COLS))

print 'g_m :', g_m.shape, g_m.dtype

# Uncomment if you decide to clip mininimum value of images to zero
# g_valid = np.clip(g_valid, a_min = 0, a_max = None)
# g_m = np.clip(g_m, a_min = 0, a_max = None)

# ------------------------------------------------------------------
print '\nCalculate Metrics'

# Image metrics
ssim,mse,psnr, nrmse = bib_metrics.compare_all_metrics(g_valid,g_m, mean = False)

# Pixelwise Metrics
rmse_pixel,amre_pixel,mre_pixel = bib_metrics.compare_all_metrics_pixel(g_valid,g_m, mean = True)
bib_metrics.plot_metric_pixel(rmse_pixel, 'RMSE', save_path,vmin = 0, vmax = 0.4,clb_legend = 'kW m$^{-3}$')
bib_metrics.plot_metric_pixel(mre_pixel, 'MRE', save_path,vmin = -.15, vmax = .15)
bib_metrics.plot_metric_pixel(amre_pixel, 'AMRE', save_path,vmin = 0, vmax = 0.30)

# Total Power Emited Error
e_power = (np.sum(g_valid,axis = (1,2)) - np.sum(g_m, axis = (1,2)))/np.sum(g_valid,axis = (1,2))

# Check Plasma Center
R_centroid, Z_centroid  = bib_geom.get_centroid(g_valid)
R_nn_centroid, Z_nn_centroid = bib_geom.get_centroid(g_m)
e_R = np.abs(R_centroid-R_nn_centroid)*1e3
e_Z = np.abs(Z_centroid-Z_nn_centroid)*1e3

# Chi2 Values
chi2_m = geom.get_chi2(f_valid,g_m,ef_valid)

print '\nAverage Values ------------------------------'
print 'ssim: %.4f +- %.4f' % (np.mean(ssim), np.std(ssim))
print 'psnr: %.2f +- %.2f' % (np.mean(psnr), np.std(psnr))
print 'nrmse: %.2f +- %.2f' % (np.mean(nrmse), np.std(nrmse))
print 'e_power: %.2f +- %.2f' % (np.mean(e_power), np.std(e_power)) 
print 'e_R: %.2f +- %.2f' % (np.mean(e_R), np.std(e_R)) 
print 'e_Z: %.2f +- %.2f' % (np.mean(e_Z), np.std(e_Z))
print 'chi2_m: %.2f +- %.2f' % (np.mean(chi2_m), np.std(chi2_m))

#------------------------------------------------------------------
print '\nGenerating save files'

print 'Creating :', save_path + 'metrics.csv'

with open(save_path + 'metrics.csv', 'w') as csvfile:
	fieldnames = ['pulse', 't', 'ssim', 'psnr', 'nrmse', 'e_power', 'e_R', 'e_Z', 'chi2_m']
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

	writer.writeheader()
	for i in range(pulse_valid.shape[0]):
		writer.writerow({'pulse': pulse_valid[i], 't': t_valid[i] , 'ssim': ssim[i],
		 'psnr': psnr[i], 'nrmse': nrmse[i], 'e_power': e_power[i], 
		 'e_R': e_R[i], 'e_Z': e_Z[i], 'chi2_m': chi2_m[i]})

#------------------------------------------------------------------
# Debuggin needed
# sometimes COMPASS chi2 from MFR does not match 
# the one calculated with bib_geom.py at all
# it seems this happens when there is saturation of detectors
# which generates poor quality reconstructions 
# these reconstructions should no be taken into account

chi2_2 = geom.get_chi2(f_valid,g_valid,ef_valid)

from collections import Counter

if np.mean(np.abs(chi2_valid-chi2_2))>0.1:
	print '\n-----------------ATTENTION---------------------'
	print 'Original chi2 from MFR Matlab'
	print 'chi2: %.5f +- %.5f' % (np.mean(chi2_valid), np.std(chi2_valid))  
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
	problematic_pulses = pulse_valid[problematic_events]
	problems_dict = Counter(problematic_pulses)
	print '\nProblematic pulses'
	for pulse in problems_dict.keys():
		print '%i \t - %i \t events' % (int(pulse), problems_dict[pulse])


