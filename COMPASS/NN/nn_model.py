
from keras import layers
from keras import models


def build_model(filters = 15, kernel = 5, input_shape = 90, dropout_rate = 0.2):
	"""
	COMPASS SXR NN model used
	Inputs: 
		filters - number of filters of the NN (as it is also changes dense layers size)
		kernel - size of 2D convolution window
		input_shape - number of inputs to the NN
		droupout_rate - dropout rate during training time at inputs
	Outputs:
		model - keras neural network model
	"""

	# initialize sequential model, layers declared henceforth are stacked on top
	# of the previous
	model = models.Sequential()

	# dropout only works during training, keras deactivates it at testing time
	model.add(layers.Dropout(rate = dropout_rate, input_shape=(input_shape,)))

	# first 2 Fully Connected (Dense Layers)
	model.add(layers.Dense(17*13*filters, activation = 'relu',))

	model.add(layers.Dense(17*13*filters, activation = 'relu'))

	# a reshape to a 3D volume
	model.add(layers.Reshape((filters,17,13)))

	# 3 2D Transposed Convolution Layers 
	model.add(layers.Conv2DTranspose(filters = filters, kernel_size = (kernel,kernel), strides = (2,2), padding = 'same',
		activation = 'relu'))

	model.add(layers.Conv2DTranspose(filters = filters, kernel_size = (kernel,kernel), strides = (2,2), padding = 'same',
		activation = 'relu'))

	model.add(layers.Conv2DTranspose(filters = filters, kernel_size = (kernel,kernel), strides = (2,2), padding = 'same',
		activation = 'relu'))

	# Final Convolution is only along the filter maps dimension (channel dimmension)
	model.add(layers.Conv2DTranspose(filters = 1, kernel_size = (1,1), strides = (1,1), padding = 'same',
		activation = 'relu'))

	# print NN model to terminal
	model.summary()

	return model