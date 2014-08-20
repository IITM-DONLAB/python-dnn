import cPickle,gzip,os,sys
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from layers.cnn import ConvLayer
from layers.logistic_sgd import LogisticRegression
from layers.mlp import HiddenLayer
from  theano.compat.python2x import OrderedDict

from models import nnet

import logging
logger = logging.getLogger(__name__)

class CNN(nnet):
	""" Instantiation of Convolution neural network ... """
	def __init__(self, numpy_rng, theano_rng, batch_size, n_outs,conv_layer_configs, hidden_layers_sizes, 
			use_fast=False,conv_activation = T.nnet.sigmoid,hidden_activation = T.nnet.sigmoid):

		super(CNN, self).__init__()
		self.layers = []

		#self.sparsity = sparsity
		#self.sparsity_weight = sparsity_weight
		#self.sparse_layer = sparse_layer

		if not theano_rng:	#if theano range not passed creating new random stream object
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

		self.x = T.ftensor4('x')  
		self.y = T.ivector('y')

		self.conv_layer_num = len(conv_layer_configs) 	#counting number of convolution layers
        	self.hidden_layer_num = len(hidden_layers_sizes)
		self.conv_layers = []
		
		logger.info('Building convolution layers....')
		for i in xrange(self.conv_layer_num):		# construct the convolution layer
			if i == 0:  				#is_input layer
				input = self.x
				is_input_layer = True
			else:
				input = self.layers[-1].output #output of previous layer
				is_input_layer = False
			config = conv_layer_configs[i]
	
			conv_layer = ConvLayer(numpy_rng=numpy_rng, input=input,input_shape=config['input_shape'],
				filter_shape=config['filter_shape'],poolsize=config['poolsize'],
				flatten = config['flatten'],activation = conv_activation, use_fast = use_fast)
			self.layers.append(conv_layer)
			self.conv_layers.append(conv_layer)
			if config['update']==True:	# only few layers of convolution layer are considered for updation
				self.params.extend(conv_layer.params)
				self.delta_params.extend(conv_layer.delta_params)

		
		self.conv_output_dim = config['output_shape'][1] * config['output_shape'][2] * config['output_shape'][3]

		logger.info('Building Hidden layers....')
		for i in xrange(self.hidden_layer_num):		# construct the hidden layer
			if i == 0:				# is first sigmoidla layer
				input_size = self.conv_output_dim
			else:
				input_size = hidden_layers_sizes[i - 1]	# number of hidden neurons in previous layers
			layer_input = self.layers[-1].output
			sigmoid_layer = HiddenLayer(rng=numpy_rng, input=layer_input,n_in=input_size, 
						n_out = hidden_layers_sizes[i], activation=hidden_activation)
			self.layers.append(sigmoid_layer)

			if config['update']==True:	# only few layers of hidden layer are considered for updation
                		self.params.extend(sigmoid_layer.params)
                		self.delta_params.extend(sigmoid_layer.delta_params)
           
		logger.info('Building last logistic layer ....')
		self.logLayer = LogisticRegression(input=self.layers[-1].output,n_in=hidden_layers_sizes[-1],n_out=n_outs)
		
		self.layers.append(self.logLayer)
	        self.params.extend(self.logLayer.params)
	        self.delta_params.extend(self.logLayer.delta_params)
		
		self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

		self.errors = self.logLayer.errors(self.y)

	"Getting CNN Feats Outputs"
	def build_out_function(self):
		feat = T.tensor4('feat')
		out_da = theano.function([feat], self.conv_layers[-1].output, updates = None, givens={self.x:feat}, on_unused_input='warn')
        	return out_da
	
	"Building fine tuning operation "
	def build_finetune_functions(self, train_shared_xy, valid_shared_xy, batch_size):

		(train_set_x, train_set_y) = train_shared_xy
		(valid_set_x, valid_set_y) = valid_shared_xy

		index = T.lscalar('index')  # index to a [mini]batch
		learning_rate = T.fscalar('learning_rate')
		momentum = T.fscalar('momentum')

		# compute the gradients with respect to the model parameters
		gparams = T.grad(self.finetune_cost, self.params)

		# compute list of fine-tuning updates
        	updates = OrderedDict()

		for dparam, gparam in zip(self.delta_params, gparams):
			updates[dparam] = momentum * dparam - gparam*learning_rate

		for dparam, param in zip(self.delta_params, self.params):
			updates[param] = param + updates[dparam]
		
		train_fn = theano.function(inputs=[index, theano.Param(learning_rate, default = 0.001),
				theano.Param(momentum, default = 0.5)],outputs=self.errors, updates=updates,
				givens={self.x: train_set_x[index * batch_size:(index + 1) * batch_size],
					self.y: train_set_y[index * batch_size:(index + 1) * batch_size]})

		valid_fn = theano.function(inputs=[index, theano.Param(learning_rate, default = 0.001),
				theano.Param(momentum, default = 0.5)],outputs=self.errors, updates=updates,
				givens={self.x: valid_set_x[index * batch_size:(index + 1) * batch_size],
					self.y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

		return train_fn, valid_fn

