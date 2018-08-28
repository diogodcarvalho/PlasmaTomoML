
import os
import sys
import numpy as np 

sys.path.insert(0, '../bib/')
import bib_data
import bib_geom
import bib_utils

#--------------------------------------------------------------------------
# Variables meaning
#
# f 		- input signal from detectors (#time_steps, #detectors)
# g     	- reconstructions (#time_steps, #pixels)
# ef 		- expected error of detectors (#time_steps, #detectors)
# fv     	- virtual signals from detectors after reconstruction
# 			  calculated by Matlab code (#time_steps, #detectors)
# t 		- time vector (#time_steps)
# chi2      - chi^2 values (~1) (#time_steps)
# pulse 	- array with pulses (#time_steps)
# SXRA 		- list of detectors used 
# SXRB		- ""
# SXRF		- ""
#
# The same logic is valid for other files in this directory
#--------------------------------------------------------------------------

# -------------------------------------------------------------------------
print '\nLoad MFR data - training/validation (OLD GEOMETRY)'

# Load Reconstructions from *.mat files, choose the correct directory
data_directory = '../data/Reconstructions/'
f_old,g_old,ef_old,fv_old,t_old,chi2_old,pulse_old,SXRA_old,SXRB_old,SXRF_old = bib_data.get_tomo_MFR(data_directory, subsampling = 1,  flatten = False)

# enforce detector shape to be always (bib_geom.N_LOS_MAX,)
f_old = np.pad(f_old,((0,0),(0,bib_geom.N_LOS_MAX-f_old.shape[1])),'constant', constant_values= 0)
ef_old = np.pad(ef_old,((0,0),(0,bib_geom.N_LOS_MAX-ef_old.shape[1])),'constant', constant_values= 0)
fv_old = np.pad(fv_old,((0,0),(0,bib_geom.N_LOS_MAX-fv_old.shape[1])),'constant', constant_values= 0)

print '\nData obtained'
print 'f_old :', f_old.shape, f_old.dtype
print 'g_old :', g_old.shape, g_old.dtype
print 'ef_old :', ef_old.shape, ef_old.dtype
print 'fv_old :', fv_old.shape, fv_old.dtype
print 't_old :', t_old.shape, t_old.dtype
print 'chi2_old:', chi2_old.shape, chi2_old.dtype
print 'pulse_old :', pulse_old.shape, pulse_old.dtype

print '\nCameras used in OLD GEOMETRY'
print 'SXRA_old :', SXRA_old
print 'SXRB_old :', SXRB_old
print 'SXRF_old :', SXRF_old

import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Load information about new geometry

# if true, assumes there are available reconstructions from the new geometry and 
# will initialize the new geometry based on the camera information present
# in these files. 
# if false, you have to input the geometry you want to mimic. Note this option
# does not generate a test set so all files who run based on the existence of a
# test set (calc_metrics.py, calc_metrics_dropout.py, plot_comparison.py)
# will not run 
test_set_available = True

import bib_data_old

if test_set_available:
	print '\nLoad MFR data - testing (NEW GEOMETRY)'

	# Load Reconstructions from *.mat files, choose the correct directory
	data_directory = '../data/Reconstructions_test/'
	f_new,g_new,ef_new,fv_new,t_new,chi2_new,pulse_new,SXRA_new,SXRB_new,SXRF_new = bib_data.get_tomo_MFR(data_directory, subsampling = 1,  flatten = False)

	# f_new,g_new,ef_new,fv_new,t_new,chi2_new,pulse_new,SXRA_new,SXRB_new,SXRF_new = get_tomo_COMPASS('../data/')
	# enforce detector shape to be always (bib_geom.N_LOS_MAX,)
	f_new = np.pad(f_new,((0,0),(0,bib_geom.N_LOS_MAX-f_new.shape[1])),'constant', constant_values= 0)
	ef_new = np.pad(ef_new,((0,0),(0,bib_geom.N_LOS_MAX-ef_new.shape[1])),'constant', constant_values= 0)
	fv_new = np.pad(fv_new,((0,0),(0,bib_geom.N_LOS_MAX-fv_new.shape[1])),'constant', constant_values= 0)

	print '\nData obtained'
	print 'f_new :', f_new.shape, f_new.dtype
	print 'g_new :', g_new.shape, g_new.dtype
	print 'ef_new :', ef_new.shape, ef_new.dtype
	print 'fv_new :', fv_new.shape, fv_new.dtype
	print 't_new :', t_new.shape, t_new.dtype
	print 'chi2_new:', chi2_new.shape, chi2_new.dtype
	print 'pulse_new :', pulse_new.shape, pulse_new.dtype

	print '\nCameras used in NEW GEOMETRY'
	print 'SXRA_new :', SXRA_new
	print 'SXRB_new :', SXRB_new
	print 'SXRF_new :', SXRF_new

	last_pulse = pulse_new[-1]

else:
	# Choose the geometry you will want to mimic
	GEOM = 201701
	last_pulse = eval('bib_geom.FIRST_SHOT_GEOM'+str(GEOM))
	print last_pulse
	SXRA_new = np.asarray(range(18,21) + range(22,35))
	SXRB_new = np.asarray(range(8,10) + [11,14] + range(16,35))+35
	SXRF_new = np.asarray([])

	print '\nCameras used in NEW GEOMETRY (user defined)'
	print 'SXRA_new :', SXRA_new
	print 'SXRB_new :', SXRB_new
	print 'SXRF_new :', SXRF_new

# -------------------------------------------------------------------------
print '\nGenerating save files'

# Directory to which all results relative to the NN training will be saved
# Be aware that changing it means you must do so in all subsequent files
save_path = './Results_virtual/'
if not os.path.exists(save_path):
        print 'Creating directory ', save_path
        os.makedirs(save_path)

print 'Creating :', save_path + 'tomo_GEOM.npz'
np.savez(save_path + 'tomo_GEOM.npz', SXRA = SXRA_new, SXRB = SXRB_new, SXRF = SXRF_new, last_pulse = last_pulse)

# -------------------------------------------------------------------------
print '\nCalculate virtual camera measurements'

# Calculate virtual camera measurements in the choosen new geometry 
# for reconstruction from older geometries
geom = bib_geom.Geometry(save_path + 'tomo_GEOM.npz')
f_virtual = geom.get_virtual_cameras(g_old, only_working = True, clip_zero = True)

print 'f_virtual :', f_virtual.shape, f_virtual.dtype

# -------------------------------------------------------------------------
print '\nGenerate final data set'

if test_set_available:
	f = np.concatenate((f_virtual, f_new),axis=0)
	g = np.concatenate((g_old,g_new),axis=0)
	ef = np.concatenate((ef_old,ef_new),axis=0)
	fv = np.concatenate((fv_old,fv_new),axis=0)
	t = np.concatenate((t_old,t_new),axis=0)
	chi2 = np.concatenate((chi2_old,chi2_new),axis=0)
	pulse = np.concatenate((pulse_old,pulse_new),axis=0)
else:
	f = f_virtual
	g = g_old
	ef = ef_old
	fv = fv_old
	t = t_old
	chi2 = chi2_old
	pulse = pulse_old

print 'f :', f.shape, f.dtype
print 'g :', g.shape, g.dtype
print 'ef :', ef.shape, ef.dtype
print 'fv :', fv.shape, fv.dtype
print 't :', t.shape, t.dtype
print 'chi2:', chi2.shape, chi2.dtype
print 'pulse :', pulse.shape, pulse.dtype

# -------------------------------------------------------------------------
print '\nDefine training/validation/test sets'

# training and validation set belong only to old geometry
i_train, i_valid,_ = bib_utils.divide_data(g_old.shape[0],ratio = [.9,.1],test_set = False,random = False)

# test set (if it exists) belongs only to new geometry
if test_set_available:
	i_test = np.arange(g_new.shape[0])+g_old.shape[0]
else:
	i_test = []

print '#train :', len(i_train)
print '#valid :', len(i_valid)
print '#test  :', len(i_test)

# -------------------------------------------------------------------------
print '\nGenerating save files'

print 'Creating :', save_path + 'tomo_COMPASS.npz'
np.savez(save_path + 'tomo_COMPASS.npz', f=f, g=g, ef=ef, fv=fv, t=t, chi2=chi2, pulse=pulse, 
	i_train = i_train, i_valid = i_valid, i_test = i_test) 






