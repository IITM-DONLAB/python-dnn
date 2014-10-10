import cPickle,gzip,os,sys,time

import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
#from theano.tensor.nnet import conv
import theano.tensor.nnet.conv3d2d

from utils.utils import activation_to_txt

T = theano.tensor
floatX = theano.config.floatX


class ConvLayer(object):
	"""Pool Layer of a convolutional network """
	def __init__(self, numpy_rng, input, input_shape, filter_shape, poolsize, activation, 
			W=None, b=None, border_mode = 'valid'):
		"""
			input_shape = (batchsize, in_time, in_channels, in_height, in_width)
			filter_shape= (flt_channels, flt_time, in_channels, flt_height, flt_width)
		"""
		assert input_shape[2] == filter_shape[2] #in_channels
		self.input = input
	
		self.input_shape = input_shape
		self.filter_shape = filter_shape
		self.poolsize = poolsize

		self.activation = activation
	
		if W is None: 		#initialize weights with random weights
			
			if activation_to_txt(activation) in ('relu','softplus'):
				norm_scale = 0.01
			else :
				fan_in = numpy.prod(filter_shape[1:]) # frames*channels*height*width
				norm_scale = 2. * numpy.sqrt( 1. / fan_in )

			initial_W = numpy.asarray(
				numpy_rng.normal(loc=0, scale=norm_scale, size=filter_shape),
				dtype=theano.config.floatX)
			W = theano.shared(value = initial_W, name = 'W')

		self.W = W

		if b is None:	# the bias is a 1D tensor -- one bias per output feature map
			if activation_to_txt(activation) in ('relu','softplus'):
				b_values = numpy.ones((filter_shape[0],), dtype=theano.config.floatX)
			else:
				b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b')
		self.b = b

		#Will be used for computing momentum
		self.delta_W = theano.shared(value = numpy.zeros(filter_shape,dtype=theano.config.floatX),
			name='delta_W')
		self.delta_b = theano.shared(value = numpy.zeros_like(self.b.get_value(borrow=True),
			dtype=theano.config.floatX), name='delta_b')

		conv_out = T.nnet.conv3d2d.conv3d(
			signals=self.input,  # Ns, Ts, C, Hs, Ws
			filters=self.W, # Nf, Tf, C, Hf, Wf
			signals_shape=input_shape,
			filters_shape=filter_shape,
			border_mode= border_mode);

		y_out = activation(conv_out + self.b.dimshuffle('x','x',0,'x','x'))
		# downsample each feature map individually, using maxpooling
		pooled_out = downsample.max_pool_2d(input=y_out, ds=poolsize, ignore_border=True)

		self.output = pooled_out

		#if flatten:		#if final convolution layer we need to flatten
		#	self.output = self.output.flatten(2)

		self.params = [self.W, self.b]
		self.delta_params = [self.delta_W, self.delta_b]
