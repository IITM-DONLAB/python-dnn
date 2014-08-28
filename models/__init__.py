import numpy
from collections import OrderedDict
import theano
import theano.tensor as T

class nnet(object):
	"""Abstract class for all Network Models"""
	def __init__(self):
		self.finetune_cost = None
		self.params = [];
		self.delta_params = [];
		self.n_layers = 0;
		self.type = None;

		# allocate symbolic variables for the data
		self.x = T.matrix('x')  # the data is presented as rasterized images
		self.y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

		#placeholders
		self.output = None
		self.features = None
		self.features_dim = None
		self.errors = None
		self.finetune_cost = None

	def getType(self):
		return self.type

	def pretraining_functions(self, train_x, batch_size):
		"""
		Should be implemeted by derived class
		"""
		raise  NotImplementedError;

	#"Building fine tuning operation "
	def build_finetune_functions(self, train_shared_xy, valid_shared_xy, batch_size):
		"""
		Generates a function `train` that implements one step of
		finetuning and a function `validate` that computes the error on 
		a batch from the validation set 

		:type train_shared_xy: pairs of theano.tensor.TensorType
		:param train_shared_xy: It is a list that contain all the train dataset, 
			pair is formed of two Theano variables, one for the datapoints,
			the other for the labels

		:type valid_shared_xy: pairs of theano.tensor.TensorType
		:param valid_shared_xy: It is a list that contain all the valid dataset, 
			pair is formed of two Theano variables, one for the datapoints,
			the other for the labels

		:type batch_size: int
		:param batch_size: size of a minibatch

        :returns (theano.function,theano.function)
		* A function for training takes minibatch_index,learning_rate,momentum 
		which updates weights,and return error rate
		* A function for validation takes minibatch_indexand return error rate
		
        """

		(train_set_x, train_set_y) = train_shared_xy
		(valid_set_x, valid_set_y) = valid_shared_xy

		index = T.lscalar('index')  # index to a [mini]batch
		learning_rate = T.scalar('learning_rate',dtype=theano.config.floatX)
		momentum = T.scalar('momentum',dtype=theano.config.floatX)

		# compute the gradients with respect to the model parameters
		gparams = T.grad(self.finetune_cost, self.params)

		# compute list of fine-tuning updates
		updates = OrderedDict()

		for dparam, gparam in zip(self.delta_params, gparams):
			updates[dparam] = momentum * dparam - gparam*learning_rate

		for dparam, param in zip(self.delta_params, self.params):
			updates[param] = param + updates[dparam]
		
		train_inputs = [index, theano.Param(learning_rate, default = 0.001),
			theano.Param(momentum, default = 0.5)]

		train_fn = theano.function(inputs=train_inputs,
			outputs=self.errors,
			updates=updates,
			givens={
				self.x: train_set_x[index * batch_size:(index + 1) * batch_size],
				self.y: train_set_y[index * batch_size:(index + 1) * batch_size]},
			allow_input_downcast=True);

		valid_fn = theano.function(inputs=[index],
			outputs=self.errors,
			givens={
				self.x: valid_set_x[index * batch_size:(index + 1) * batch_size],
				self.y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

		return train_fn, valid_fn

	def build_test_function(self,test_shared_xy,batch_size):
		"""
		Get Fuction for testing

		:type test_shared_xy: pairs of theano.tensor.TensorType
		:param test_shared_xy: It is a list that contain all the test dataset, 
			pair is formed of two Theano variables, one for the datapoints,
			the other for the labels

		:type batch_size: int
		:param batch_size: size of a minibatch
				
		:returns theano.function
		A function which takes index to minibatch and Generates Label Array and error

		"""
		(test_set_x, test_set_y) = test_shared_xy
		index = T.lscalar('index')  # index to a [mini]batch
		test_fn = theano.function(inputs=[index],
			outputs=[self.output ,self.errors],
			givens={
				self.x: test_set_x[index * batch_size:(index + 1) * batch_size],
				self.y: test_set_y[index * batch_size:(index + 1) * batch_size]})
		return test_fn
	
	def getFeaturesFunction(self):
		"""
		Get Function for extracting Feature/Bottle neck
				
		:returns theano.function
		A function takes input features 
		"""
		#in_x = T.matrix('in_x');
		in_x = self.x.type('in_x');
		fn = theano.function(inputs=[in_x],outputs=self.features,
			givens={self.x: in_x},name='features')#,on_unused_input='warn')
		return fn
