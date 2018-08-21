
import numpy as np
np.random.seed(0)

from keras.models import *
from keras.layers.core import *
from keras.layers.convolutional import *


def build_model(filters = 20, dropout_rate = 0.):
	"""
	JET Bolometer NN model used
	Inputs: 
		filters - number of filters of NN
		droupout_rate - dropout rate during training time at inputs
	Outputs:
		model - neural network model
	"""

	inputs = Input(shape=(56,))

	x = Dropout(dropout_rate)(inputs)

	x = Dense(15*25*filters)(x)
	x = Activation('relu')(x)

	x = Dense(15*25*filters)(x)
	x = Activation('relu')(x)

	x = Reshape((filters,15,25))(x)

	x = Conv2DTranspose(filters, kernel_size=(5,5), strides=(2,2), padding='same', data_format='channels_first')(x)
	x = Activation('relu')(x)

	x = Conv2DTranspose(filters, kernel_size=(5,5), strides=(2,2), padding='same', data_format='channels_first')(x)
	x = Activation('relu')(x)

	x = Conv2DTranspose(filters, kernel_size=(5,5), strides=(2,2), padding='same', data_format='channels_first')(x)
	x = Activation('relu')(x)

	outputs = Conv2DTranspose(1, kernel_size=(5,5), strides=(1,1), padding='same', data_format='channels_first')(x)

	model = Model(inputs=inputs, outputs=outputs)

	print '%-10s %-20s %-20s %-20s' % ('layer', 'class_name', 'input_shape', 'output_shape')
	print '-'*73
	for i, layer in enumerate(model.layers):
	    class_name = layer.__class__.__name__
	    input_shape = layer.input_shape[1:]
	    output_shape = layer.output_shape[1:]
	    print '%-10d %-20s %-20s %-20s' % (i+1, class_name, str(input_shape), str(output_shape))
	print '-'*73

	return model

