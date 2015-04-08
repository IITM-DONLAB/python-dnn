import cPickle, gzip, os, sys, time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class HiddenLayer(object):
	""" Class for hidden layer """
	def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh, 
		adv_activation_method = None, pool_size = 1, pnorm_order = 1):

		self.input = input
		self.n_in = n_in
		self.n_out = n_out

		if W is None:
			W_bound = -numpy.sqrt(6. / (n_in + n_out))
			W_values = numpy.asarray(rng.uniform(low=-W_bound,high=W_bound,
					size=(n_in, n_out)), dtype=theano.config.floatX)
			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4
			W = theano.shared(value=W_values, name='W', borrow=True)
		self.W = W

		if b is None:
			b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)
		self.b = b
		
		#Will be used for computing momentum
		self.delta_W = theano.shared(value = numpy.zeros((n_in,n_out),dtype=theano.config.floatX), name='delta_W')
		self.delta_b = theano.shared(value = numpy.zeros_like(self.b.get_value(borrow=True), dtype=theano.config.floatX), name='delta_b')

		self.lin_output = T.dot(input, self.W) + self.b
	
		if adv_activation_method == 'maxout':	# pooling of output of neuron based on poolsize
			self.last_start = n_out - pool_size
			self.tmp_output = self.lin_output[:,0:self.last_start+1:pool_size]
			for i in range(1, pool_size):
				cur = self.lin_output[:,i:self.last_start+i+1:pool_size]
				self.tmp_output = T.maximum(cur, self.tmp_output)
			self.output = activation(self.tmp_output)
		elif adv_activation_method == 'pnorm': # pooling of output of neuron based on poolsize and normalizing the output
			self.last_start = n_out - pool_size
			self.tmp_output = abs(self.lin_output[:,0:self.last_start+1:pool_size]) ** pnorm_order
			for i in range(1, pool_size):
				cur = abs(self.lin_output[:,i:self.last_start+i+1:pool_size]) ** pnorm_order
				self.tmp_output = self.tmp_output + cur
			self.tmp_output = self.tmp_output ** (1.0 / pnorm_order)
			self.output = activation(self.tmp_output)
		else:
			self.output = (self.lin_output if activation is None
						   else activation(self.lin_output))

		# parameters of the model
		self.params = [self.W, self.b]
		self.delta_params = [self.delta_W, self.delta_b]


class DropoutHiddenLayer(HiddenLayer):
	def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh,
			adv_activation_method = None, pool_size = 1,pnorm_order=1,dropout_factor=0.5):
		super(DropoutHiddenLayer, self).__init__(rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
				activation=activation, adv_activation_method = adv_activation_method, pool_size = pool_size,pnorm_order=pnorm_order)
		self.theano_rng = RandomStreams(rng.randint(2 ** 30))
		dropout_prob = self.theano_rng.binomial(n=1, p=1-dropout_factor, size=self.output.shape,
			dtype=theano.config.floatX)	
		self.dropout_output = dropout_prob * self.output

def _dropout_from_layer(theano_rng, hid_out, p):
    """ p is the factor for dropping a unit """
    # p=1-p because 1's indicate keep and p is prob of dropping
    return theano_rng.binomial(n=1, p=1-p, size=hid_out.shape,dtype=theano.config.floatX) * hid_out

