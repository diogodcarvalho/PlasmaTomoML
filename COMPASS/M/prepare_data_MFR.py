
import os
import sys
import numpy as np 

sys.path.insert(0, '../bib/')
import bib_data
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
print '\nLoad MFR data'

# Load Reconstructions from *.mat files generated with MFR Matlab code, choose the correct directory
data_directory = '../data/Reconstructions/'
f,g,ef,fv,t,chi2,pulse,SXRA,SXRB,SXRF = bib_data.get_tomo_MFR(data_directory, subsampling = 1,  flatten = True)

print '\nData obtained'
print 'f :', f.shape, f.dtype
print 'g :', g.shape, g.dtype
print 'ef :', ef.shape, ef.dtype
print 'fv :', fv.shape, fv.shape
print 't :', t.shape, t.dtype
print 'chi2:', chi2.shape, chi2.dtype
print 'pulse :', pulse.shape, pulse.dtype

print '\nCameras used'
print 'SXRA :', SXRA
print 'SXRB :', SXRB
print 'SXRF :', SXRF

# -------------------------------------------------------------------------
print '\nDefine training/validation/test sets'

i_train, i_valid, _ = bib_utils.divide_data(g.shape[0],ratio = [.9,.1],test_set = False,random = False)

print '#train :', len(i_train)
print '#valid :', len(i_valid)

# -------------------------------------------------------------------------
print '\nGenerating save files'

# Directory to which all results relative to the matrix fitting will be saved
# Be aware that changing this directory here, means you must do so in all subsequent files
save_path = './Results/'
if not os.path.exists(save_path):
        print 'Creating directory ', save_path
        os.makedirs(save_path)

print 'Creating :', save_path + 'tomo_COMPASS.npz'
np.savez(save_path + 'tomo_COMPASS.npz', f=f, g=g, ef=ef, fv=fv, t=t, chi2=chi2, pulse=pulse, 
	i_train = i_train, i_valid = i_valid) 

print 'Creating :', save_path + 'tomo_GEOM.npz'
np.savez(save_path + 'tomo_GEOM.npz', SXRA = SXRA, SXRB = SXRB, SXRF = SXRF, last_pulse = pulse[-1])
