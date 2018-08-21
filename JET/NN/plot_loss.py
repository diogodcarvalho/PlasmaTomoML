
import numpy as np
import matplotlib.pyplot as plt 

# ----------------------------------------------------------
# Load log file with training/validation loss values

save_path = './Results/'
fname = save_path + 'train.log'
loadtxt = np.loadtxt(fname,skiprows = 1,usecols = (2,3,4))
print loadtxt.shape

i = loadtxt[:,0]
loss_train = loadtxt[:,1]*10**3
loss_valid = loadtxt[:,2]*10**3

# ----------------------------------------------------------
# Plot loss function

plt.figure()
plt.plot(i,loss_train,label = 'training')
plt.plot(i,loss_valid,label = 'validation')
plt.plot((np.argmin(loss_valid),np.argmin(loss_valid)),(0,np.min(loss_valid)),'k--',label='min validation loss')
plt.xlabel('# epoch')
plt.ylabel('$\mathcal{L}$ (kW/m$^3$)')
plt.ylim([0,np.max(loss_train)+2])
plt.grid(True)
plt.legend()
plt.savefig(save_path + 'loss_log.png',dpi = 300, bbox_inches='tight')
plt.show()