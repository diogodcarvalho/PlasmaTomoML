
import time
import numpy as np

# -------------------------------------------------------------------------
print '\nLoad data'

save_path = './Results/'
tomo_COMPASS = np.load(save_path + 'tomo_COMPASS.npz')
f = tomo_COMPASS['f']
g = tomo_COMPASS['g']

f = f.transpose()
g = g.transpose()

print 'g:', g.shape, g.dtype
print 'f:', f.shape, f.dtype

# -------------------------------------------------------------------------
print '\nDefine training set'

f_train = f[:,tomo_COMPASS['i_train']]
g_train = g[:,tomo_COMPASS['i_train']]

print 'f_train:', f_train.shape
print 'g_train:', g_train.shape

# ------------------------------------------------------------------------
# Initialize matrix with zeros

M = np.zeros((g.shape[0],f.shape[0]), dtype=np.float32)
print 'M:', M.shape, M.dtype

# -------------------------------------------------------------------------
print '\nInitialize theano variables'

import theano
import theano.tensor as T
from theano.printing import pydotprint

g = theano.shared(g, 'g')
f = theano.shared(f, 'f')

M = theano.shared(M, 'M')
loss = T.mean(T.abs_(T.dot(M,f)-g))
grad = T.grad(loss, M)

# -------------------------------------------------------------------------
print '\nInitialize Gradient Descent'

# the values of the learning rate, momentum and epochs might need to be
# adjusted for different data-sets for a better convergence

learning_rate = np.float32(1.)
momentum = np.float32(0.9)

print 'learning_rate:', learning_rate
print 'momentum:', momentum

updates = []

m = theano.shared(M.get_value() * np.float32(0.))
v = momentum * m - learning_rate * grad

updates.append((m, v))
updates.append((M, M + momentum * v - learning_rate * grad))

train = theano.function(inputs=[],
                        outputs=[loss],
                        updates=updates)

pydotprint(train, outfile= save_path + 'train.png', compact=False)

# -------------------------------------------------------------------------
print '\nRun Gradien Descent'

epochs = int(1e5)

# saves the loss functions values to a *.log file
fname = save_path + 'train.log'
print 'Writing:', fname
f = open(fname, 'w')

# training can be interrupted at any time by pressing ^c
# all logs until that point and curret matrix obtained will be saved
try:
    for epoch in range(epochs):
        if epoch == 0:
            s = '%-10s %10s %10s' % ('time', 'epoch', 'loss (kW/m^3)')
            print s
            f.write(s)
            f.write('\n')
            f.flush()
        outputs = train()
        loss_value = outputs[0]
        t = time.strftime('%H:%M:%S')
        s = '%-10s %10d %10.6f' % (t, epoch, loss_value)
        if epoch%100 == 0:
            print s
        f.write(s)
        f.write('\n')
        f.flush()

except KeyboardInterrupt:
    print 'Training interrupted.'

f.close()

# -------------------------------------------------------------------------
print '\n Save matrix obtained'

M = M.get_value()

print 'M:', M.shape, M.dtype

fname = save_path + 'M.npy'
print 'Writing:', fname
np.save(fname, M)


    

