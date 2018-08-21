
import sys
import time
from keras.callbacks import *

def write(text,fname,reset = False):
    """
    Auxiliar function to print loss functions in log file and terminal
    Inputs:
        text - text to print
        reset - if True resets *.log file
        fname - path to log file
    Outputs:
        None
    """
    sys.stdout.write(text)
    sys.stdout.flush()
    fname = fname + '.log'
    if reset:
        f = open(fname, 'w')
    else:
        f = open(fname, 'a')
    f.write(text)
    f.close()

class MyCallback(Callback):
    """
    Callback class defined for the NN training

    Attributes:
        min_val_loss - minimum registered validation loss value
        save_path - directory in which log file where loss values will be stored
    """
    def __init__(self,save_path):
        '''
        Inputs:
            fname -  - path to log file where loss values are stored
        '''
        self.min_val_loss = None
        self.save_path = save_path
        write('%-20s %5s %10s %10s\n' % ('time', 'epoch', 'loss', 'val_loss'), self.save_path + 'train',True)
    def on_epoch_end(self, epoch, logs={}):
        '''
        Function called at the end of each epoch. Action performed depends if validation loss value obtained
        is a new minimum. If so the correspondent NN parameters are saved model_parameters.hdf. Otherwise only 
        the train.log file will be updated with the loss values obtained
        '''
        epoch += 1
        loss = logs['loss']
        val_loss = logs['val_loss']
        t = time.strftime('%Y-%m-%d %H:%M:%S')
        if (self.min_val_loss == None) or (val_loss < self.min_val_loss):
            self.min_val_loss = val_loss
            write('%-20s %5d %10.4f %10.4f *\n' % (t, epoch, loss, val_loss), self.save_path + 'train')
            if epoch >= 100:
                self.model.save_weights(self.save_path + 'model_parameters.hdf', overwrite=True)
        else:
            write('%-20s %5d %10.4f %10.4f\n' % (t, epoch, loss, val_loss), self.save_path + 'train')