
import numpy as np
import matplotlib.pyplot as plt 

# -------------------------------------------------------------------------
# Load log file with training loss values

save_path = './Results/'
fname = save_path + 'train.log'
loadtxt = np.loadtxt(fname,skiprows = 1,usecols = (1,2))
print loadtxt.shape

i = loadtxt[:,0]
loss_train = loadtxt[:,1]

# -------------------------------------------------------------------------
# Plot loss function

print 'Creating :', save_path + 'loss_log.png'

plt.figure()
plt.plot(i/1e3,loss_train,label = 'training')
plt.xlabel('# epoch (10$^3$)')
plt.ylabel('$\mathcal{L}$ (kW/m$^3$)')
plt.ylim([0,np.max(loss_train)*1.05])
plt.grid(True)
plt.legend()
plt.savefig(save_path + 'loss_log.png',dpi = 300, bbox_inches='tight')
plt.show()