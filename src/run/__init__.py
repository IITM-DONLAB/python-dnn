import time,numpy,os
import logging
logger = logging.getLogger(__name__)

from io_modules.file_reader import read_dataset
from utils.learn_rates import LearningRate
from io_modules.data_exporter import export_data

def _testing(nnetModel,test_sets, test_xy, test_x, test_y,batch_size):

	# get the testing function for the model
	logger.info('Getting the Test function')
	test_fn = nnetModel.build_test_function((test_x, test_y), batch_size=batch_size)

	logger.info('Starting Testing');
	test_error  = []
	while not test_sets.is_finish():
		test_sets.make_partition_shared(test_xy)
		n_test_batches= test_sets.cur_frame_num / batch_size;
		test_losses = [test_fn(i) for i in xrange(n_test_batches)]
		test_error.extend(test_losses)
		test_sets.read_next_partition_data()
		logger.debug("Test Error (upto curr part) = %f",numpy.mean(test_error))


	test_error=numpy.mean(test_error)
	logger.info('Optimization complete with best Test score of %f %%',test_error * 100)

	return test_error;

def testing(nnetModel,model_config,data_spec):
	batch_size = model_config['batch_size']
	try:
		test_sets, test_xy, test_x, test_y = read_dataset(data_spec['testing']) 
	except KeyError:
		#raise e
		logger.info("No testing set:Skiping Testing");
	else:
		_testing(nnetModel,test_sets, test_xy, test_x, test_y,batch_size)


def _fineTunning(nnetModel,train_sets,train_xy,train_x,train_y,
		valid_sets,valid_xy,valid_x,valid_y,lrate,momentum,batch_size):

	def valid_score():
		valid_error = []
		while not valid_sets.is_finish():
			valid_sets.make_partition_shared(valid_xy)
			n_valid_batches= valid_sets.cur_frame_num / batch_size;
			validation_losses = [validate_fn(i) for i in xrange(n_valid_batches)]
			valid_error.extend(validation_losses)
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

def fineTunning(nnetModel,model_config,data_spec):
	batch_size = model_config['batch_size']
	try:
		train_sets, train_xy, train_x, train_y = read_dataset(data_spec['training'])
		valid_sets, valid_xy, valid_x, valid_y = read_dataset(data_spec['validation'])
	except KeyError:
		#raise e
		logger.info("No validation/training set:Skiping Fine tunning");
	else:
		try:
			finetune_method = model_config['finetune_method']
			finetune_config = model_config['finetune_rate']
			momentum = model_config['finetune_momentum']
			lrate = LearningRate.get_instance(finetune_method,finetune_config);
		except KeyError, e:
			print("KeyMissing:"+str(e));
			print("Fine tunning Paramters Missing")
			sys.exit(2)


		_fineTunning(nnetModel,train_sets,train_xy,train_x,train_y,
			valid_sets,valid_xy,valid_x,valid_y,lrate,momentum,batch_size)


def exportFeatures(nnetModel,model_config,data_spec):
	try:
		export_path = model_config['export_path']
		data_spec_testing =data_spec['testing']
	except KeyError:
		#raise e
		logger.info("No testing set/export path:Skiping Exporting");
	else :
		out_function = nnetModel.getFeaturesFunction()
		export_data(data_spec_testing,export_path,out_function,nnetModel.features_dim);


def saveLabels(nnetModel,export_path,data_spec):

	getLabel = nnetModel.getLabelFunction()
	test_sets  = read_dataset(in_child_options,pad_zeros=True)[0]
	while (not test_sets.is_finish()):
		for batch_index in xrange(test_sets.cur_frame_num/batch_size):
			s_idx = batch_index*batch_size; e_idx = s_idx + batch_size
		
			pred = getLabel(test_sets.feat[s_idx:e_idx])
		
			e_idx= min(test_sets.cur_frame_num -test_sets.num_pad_frames,s_idx+batch_size);
		test_sets.read_next_partition_data(pad_zeros=True);

def createDir(wdir):
	"""create working dir"""
	wdir = os.path.abspath(wdir)
	logger.info("Creating working dir %s ...",wdir)
	if not os.path.exists(wdir):
		try:
			os.makedirs(wdir)
			logger.info("Creating working dir ... DONE");
		except Exception, e:
			raise e
	else:
		logger.info("Creating working dir  ... Skipping");
