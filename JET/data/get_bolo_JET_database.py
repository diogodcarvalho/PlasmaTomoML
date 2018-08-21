
# -------------------------------------------------------------------------------
# ATTENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# The following script works only inside a JET cluster
# Output *.hdf files will be generated so one can work offline afterwards

import h5py
import numpy as np
from ppf import *

ppfgo()
ppfssr(i=[0,1,2,3,4])
ppfuid('jetppf', 'R')

# -------------------------------------------------------------------------------
# Auxiliary functions

def get_data(pulse, camera):
    ihdata, iwdata, data, x, t, ier = ppfget(pulse, 'bolo', camera, reshape=1)
    if ier != 0:
        print 'Error:', ier
        exit()
    return data, t

def get_kb5(pulse):
    kb5v, kb5v_t = get_data(pulse, 'kb5v')
    kb5h, kb5h_t = get_data(pulse, 'kb5h')
    assert np.all(kb5v_t == kb5h_t)
    kb5 = np.hstack((kb5v, kb5h))
    kb5_t = kb5v_t
    print 'kb5:', kb5.shape, kb5.dtype
    print 'kb5_t:', kb5_t.shape, kb5_t.dtype
    return kb5, kb5_t

def trunc(t, dt):
    digits = len(str(dt).split('.')[1])
    t = float(int(t*(10.**digits)))/(10.**digits)
    return t

def get_bolo(pulse, tmin, tmax, dt):
    """
    Returns bolometer measures from the JET database for a given
    pulse and time interval
    Inputs:
        pulse - (int) pulse number
        tmin - (int) initial time
        tmax - (int) final time
        dt - (int) timestep
    Outputs:
        bolo - (array) bolometer measures
        bolo_t - (array) time vector associated
    """
    kb5, kb5_t = get_kb5(pulse)
    window = 2.*0.0025
    kb5_tmin = kb5_t[0]
    kb5_tmax = kb5_t[-1]-window
    print 'kb5_tmin:', kb5_tmin
    print 'kb5_tmax:', kb5_tmax
    if tmin < kb5_tmin:
        tmin = trunc(kb5_tmin, dt)
        print 'tmin:', tmin
    if tmax > kb5_tmax:
        tmax = trunc(kb5_tmax, dt)
        print 'tmax:', tmax
    num = int(round((tmax-tmin)/dt)) + 1
    bolo = []
    bolo_t = np.linspace(tmin, tmax, num=num, dtype=np.float32)
    for t in bolo_t:
        i0 = np.argmin(np.fabs(kb5_t - t))
        i1 = np.argmin(np.fabs(kb5_t - (kb5_t[i0] + window)))
        print '%10.4f %10d %10d %10d %10.4f %10.4f' % (t, i0, i1, i1-i0+1, kb5_t[i0], kb5_t[i1])
        mean = np.mean(kb5[i0:i1+1], axis=0)
        bolo.append(mean)
    bolo = np.array(bolo)
    print 'bolo:', bolo.shape, bolo.dtype
    print 'bolo_t:', bolo_t.shape, bolo_t.dtype
    return bolo, bolo_t

# -------------------------------------------------------------------------------
# Generate save file

fname = 'bolo_JET.hdf'
print 'Writing:', fname
f = h5py.File(fname, 'w')

# pulses which we want to recover, edit as you wish
pulses = []

# a good pulse that has nice reconstructions
# pulses = [92213]

print 'pulses:', pulses

for pulse in pulses:
    bolo, bolo_t = get_bolo(pulse, 40., 100., 0.01)
    g = f.create_group(str(pulse))
    g.create_dataset('bolo', data=bolo)
    g.create_dataset('t', data=bolo_t)

f.close()
