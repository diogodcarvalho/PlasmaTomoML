
# -------------------------------------------------------------------------------
# ATTENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# The following script works only inside a JET cluster
# Output *.hdf files will be generated so one can work offline afterwards

import h5py
import numpy as np
from ppf import *

ppfgo()
ppfssr(i=[0,1,2,3,4])

# -------------------------------------------------------------------------------
# Auxiliary functions

def get_tomo(pulse):
    """
    Returns existing tomographic reconstructions in the JET database for a given
    pulse
    Inputs:
        pulse - (int) pulse number
    Outputs:
        tomo - (array) existing reconstructions
        tomo_t - (array) correspondent time vector
    """
    ppfuid('bolom', 'r')
    tomo = []
    tomo_t = []
    for i in range(1, 100):
        pr = 'pr%02d' % i
        ihdata, iwdata, data, x, t, ier = ppfget(pulse, 'bolt', pr, reshape=1)
        if ier != 0:
            break
        t = np.float32(ihdata.strip().split()[-1][2:-1])
        tomo.append(np.flipud(np.transpose(data)))
        tomo_t.append(t)
    if len(tomo) > 0:
        tomo = np.array(tomo)
        tomo_t = np.array(tomo_t)
        print pulse, 'tomo:', tomo.shape, tomo.dtype
    return tomo, tomo_t

def get_kb5(pulse, tomo_t):
    """
    Returns bolometer measures from the JET database for a given
    pulse and time vector
    Inputs:
        pulse - (int) pulse number
        tomo_t - (array) time vector
    Outputs:
        kb5 - (array) bolometer measures
    """
    dt = 0.0025
    ppfuid('jetppf', 'r')
    ihdata, iwdata, data, x, t, ier = ppfget(pulse, 'bolo', 'kb5v', reshape=1)
    kb5v = data
    kb5v_t = t
    print pulse, 'kb5v:', kb5v.shape, kb5v.dtype
    ihdata, iwdata, data, x, t, ier = ppfget(pulse, 'bolo', 'kb5h', reshape=1)
    kb5h = data
    kb5h_t = t
    print pulse, 'kb5h:', kb5h.shape, kb5h.dtype
    assert np.all(kb5v_t == kb5h_t)
    kb5_t = kb5v_t
    kb5 = []
    for t in tomo_t:
        i0 = np.argmin(np.fabs(kb5_t - t))
        i1 = np.argmin(np.fabs(kb5_t - (t + 2.*dt)))
        print '%10d %10.4f %10d %10d %10d %10.4f %10.4f' % (len(kb5)+1, t, i0, i1, i1-i0+1, kb5_t[i0], kb5_t[i1])
        hstack = np.hstack((kb5v[i0:i1+1],kb5h[i0:i1+1]))
        mean = np.mean(hstack, axis=0)
        kb5.append(mean)
    kb5 = np.array(kb5)
    print pulse, 'kb5:', kb5.shape, kb5.dtype
    return kb5

# -------------------------------------------------------------------------------
# Generate save file

fname = 'tomo_JET.hdf'
print 'Writing:', fname
f = h5py.File(fname, 'w')

# pulses which we want to recover, edit as you wish
pulses = []
print 'pulses:', pulses

for pulse in pulses:
    tomo, tomo_t = get_tomo(pulse)
    if len(tomo) > 0:
        kb5 = get_kb5(pulse, tomo_t)
        g = f.create_group(str(pulse))
        g.create_dataset('t', data=tomo_t)
        g.create_dataset('kb5', data=kb5)
        g.create_dataset('tomo', data=tomo)

f.close()
