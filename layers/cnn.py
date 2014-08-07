import cPickle,gzip,os,sys,time

import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from theano.sandbox.cuda.basic_ops import gpu_contiguous
#from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
#from pylearn2.sandbox.cuda_convnet.pool import MaxPool

class ConvLayer(object):
	"""Pool Layer of a convolutional network """
	def __init__(self, numpy_rng, input, input_shape, filter_shape, poolsize, activation, 
			flatten = False, border_mode = 'valid', W=None, b=None, use_fast = False):
	
		assert input_shape[1] == filter_shape[1]
		self.input = input.reshape(input_shape)
	
		self.input_shape = input_shape
		self.filter_shape = filter_shape
		self.poolsize = poolsize

		self.activation = activation
		self.flatten = flatten
	
		fan_in = numpy.prod(filter_shape[1:])
		fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(poolsize))
	
		if W is None: 		#initialize weights with random weights in range (-w_bound,w_bound)
			W_bound = numpy.sqrt(6. / (fan_in + fan_out))
			initial_W = numpy.asarray(
				numpy_rng.uniform(low=-W_bound, high=W_bound,size=filter_shape),
				dtype=theano.config.floatX)

			if activation == T.nnet.sigmoid:
				initial_W *= 4
			W = theano.shared(value = initial_W, name = 'W')
		self.W = W
		
		if b is None:		# the bias is a 1D tensor -- one bias per output feature map
			b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b')
		self.b = b
	
		#Will be used for computing momentum
		self.delta_W = theano.shared(value = numpy.zeros(filter_shape,dtype=theano.config.floatX), name='delta_W')
		self.delta_b = theano.shared(value = numpy.zeros_like(self.b.get_value(borrow=True),dtype=theano.config.floatX), name='delta_b')


		if use_fast:		#uses pylearn2 modules but it has got lot of limitations
			input_shuffled = self.input.dimshuffle(1, 2, 3, 0) #rotating axes towards right
			filters_shuffled = self.W.dimshuffle(1, 2, 3, 0) #rotating axes towards right
			conv_op = FilterActs()
			contiguous_input = gpu_contiguous(input_shuffled)
			contiguous_filters = gpu_contiguous(filters_shuffled)
			conv_out_shuffled = conv_op(contiguous_input, contiguous_filters)
			y_out_shuffled = activation(conv_out_shuffled + self.b.dimshuffle(0, 'x', 'x', 'x'))
			pool_op = MaxPool(ds=poolsize[0], stride=poolsize[0])	#only supports square window for pooling
			pooled_out = pool_op(y_out_shuffled).dimshuffle(3, 0, 1, 2) # roating axes back
		else:			#uses theano modules, border_mode - ?
			conv_out = conv.conv2d(input=self.input, filters=self.W,filter_shape=filter_shape,
					image_shape=input_shape,border_mode = border_mode)
			y_out = activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
			# downsample each feature map individually, using maxpooling
			pooled_out = downsample.max_pool_2d(input=y_out, ds=poolsize, ignore_border=True)

		self.output = pooled_out

		if flatten:		#if final convolution layer we need to flatten
			self.output = self.output.flatten(2)

		self.params = [self.W, self.b]
		self.delta_params = [self.delta_W, self.delta_b]


