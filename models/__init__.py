import numpy
from collections import OrderedDict
import theano
import theano.tensor as T
import time

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
		in_x = x.type('in_x');
		fn = theano.function(inputs=[in_x],outputs=[self.features],
			givens={self.x: in_x},name='features')#,on_unused_input='warn')
		return fn


import logging
logger = logging.getLogger(__name__)

def testing(nnetModel,test_sets, test_xy, test_x, test_y,batch_size):

	# get the testing function for the model
	logger.info('Getting the Test function')
	test_fn = nnetModel.build_test_function((test_x, test_y), batch_size=batch_size)

	logger.info('Starting Testing');
	
	test_error  = []
	test_output = numpy.array([],int);
	while not test_sets.is_finish():
		test_sets.make_partition_shared(test_xy)
		n_test_batches= test_sets.cur_frame_num / batch_size;
		for i in xrange(n_test_batches):
			pred, err = test_fn(i)
			test_error.append(err)
			test_output=numpy.append(test_output,pred)
		test_sets.read_next_partition_data()
		logger.debug("Test Error (upto curr part) = %f",numpy.mean(test_error))
	test_sets.initialize_read();

	test_loss=numpy.mean(test_error)
	logger.info('Optimization complete with best Test score of %f %%',test_loss * 100)

	return test_output,test_loss;

def fineTunning(nnetModel,train_sets,train_xy,train_x,train_y,
		valid_sets,valid_xy,valid_x,valid_y,lrate,momentum,batch_size):

	def valid_score():
		valid_error = [] 
		while not valid_sets.is_finish():
			valid_sets.make_partition_shared(valid_xy)
			n_valid_batches= valid_sets.cur_frame_num / batch_size;
			validation_losses = [validate_fn(i) for i in xrange(n_valid_batches)]
			valid_error.append(validation_losses)
			valid_sets.read_next_partition_data()
			logger.debug("Valid Error (upto curr part) = %f",numpy.mean(valid_error))
		valid_sets.initialize_read();
		return numpy.mean(valid_error);

	# get the training, validation function for the model
	logger.info('Getting the finetuning functions')
	train_fn, validate_fn = nnetModel.build_finetune_functions((train_x, train_y),
			 (valid_x, valid_y), batch_size=batch_size)
		
	best_validation_loss=float('Inf')

	logger.info('Finetunning the model..');	
	start_time = time.clock()

	while (lrate.get_rate() != 0):
		train_error = []
		while not train_sets.is_finish():
			train_sets.make_partition_shared(train_xy)
			for batch_index in xrange(train_sets.cur_frame_num / batch_size):  # loop over mini-batches
				train_error.append(train_fn(index=batch_index,
					learning_rate = lrate.get_rate(), momentum = momentum))
				logger.debug('Training batch %d error %f',batch_index, numpy.mean(train_error))
			train_sets.read_next_partition_data()
		logger.info('Fine Tunning:epoch %d, training error %f',lrate.epoch, numpy.mean(train_error));
		train_sets.initialize_read()

		valid_error = valid_score()
		if valid_error < best_validation_loss:
			best_validation_loss=valid_error
		logger.info('Fine Tunning:epoch %d, validation error %f',lrate.epoch, valid_error);
		lrate.get_next_rate(current_error = 100 * valid_error)
	
	end_time = time.clock()

	logger.info('Best validation error %f',best_validation_loss)

	logger.info('The Fine tunning ran for %.2fm' % ((end_time - start_time) / 60.))
	logger.info('Optimization complete with best validation score of %f %%', best_validation_loss * 100)

	return best_validation_loss


def getFeatures(nnetModel,data_spec_testing):
	out_function = nnetModel.getFeaturesFunction()
	test_sets, test_xy, test_x, test_y = read_dataset(data_spec_testing)
	while (not test_sets.is_finish()):
		data = out_function(test_sets.feat)
		test_sets.read_next_partition_data()
		#TODO write data


