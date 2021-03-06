import time,numpy,os
import threading
import logging
logger = logging.getLogger(__name__)

from pythonDnn.io_modules.file_reader import read_dataset
from pythonDnn.utils.learn_rates import LearningRate
from pythonDnn.io_modules.data_exporter import export_data

def _testing(nnetModel,test_sets):

	test_xy = test_sets.shared_xy
	test_x = test_sets.shared_x
	test_y = test_sets.shared_y

	batch_size = test_sets.batch_size
	
	# get the testing function for the model
	logger.info('Getting the Test function')
	test_fn = nnetModel.build_test_function((test_x,test_y), batch_size=batch_size)

	logger.info('Starting Testing');
	test_error  = []
	while not test_sets.is_finish():
		test_losses = [test_fn(i) for i in xrange(test_sets.nBatches)]
		test_error.extend(test_losses)
		test_sets.read_next_partition_data()
		logger.debug("Test Error (upto curr part) = %f",numpy.mean(test_error))


	test_error=numpy.mean(test_error)
	logger.info('Optimization complete with best Test score of %f %%',test_error * 100)

	return test_error;

def testing(nnetModel,data_spec,saveLabel=True,outFile='test.out'):
	try:
		test_sets = read_dataset(data_spec['testing']) 
	except KeyError:
		#raise e
		logger.info("No testing set:Skiping Testing");
	else:
		_testing(nnetModel,test_sets)
		if saveLabel:
			saveLabels(nnetModel,outFile,data_spec['testing'])


def _fineTunning(nnetModel,train_sets,valid_sets,lrate,momentum,saveFeq=0,outFile=''):
	
	train_xy = train_sets.shared_xy
	train_x = train_sets.shared_x
	train_y = train_sets.shared_y

	valid_xy = valid_sets.shared_xy
	valid_x = valid_sets.shared_x
	valid_y = valid_sets.shared_y	

	def valid_score():
		val_batch_size = valid_sets.batch_size
		valid_error = []
		while not valid_sets.is_finish():
			validation_losses = [validate_fn(i) for i in xrange(valid_sets.nBatches)]
			valid_error.extend(validation_losses)
			logger.debug("Valid Error (upto curr part) = %f",numpy.mean(valid_error))
			valid_sets.read_next_partition_data()
		valid_sets.initialize_read();
		return numpy.mean(valid_error);

	# get the training, validation function for the model
	batch_size = train_sets.batch_size
	logger.info('Getting the finetuning functions')
	train_fn, validate_fn = nnetModel.build_finetune_functions((train_x,train_y),
		(valid_x,valid_y), batch_size=batch_size)
	
	best_validation_loss=float('Inf')

	logger.info('Finetunning the model..');
	start_time = time.clock()

	while (lrate.get_rate() != 0):
		train_error = []
		while not train_sets.is_finish():
			for batch_index in xrange(train_sets.nBatches):  # loop over mini-batches
				train_error.append(train_fn(index=batch_index,
					learning_rate = lrate.get_rate(), momentum = momentum))
				logger.debug('Training batch %d error %f',batch_index, numpy.mean(train_error))
			train_sets.read_next_partition_data()
		logger.info('Fine Tunning:epoch %d, training error %f',lrate.epoch, numpy.mean(train_error));

		#savemodel
		savethread=None

		if(saveFeq!=0 and lrate.epoch%saveFeq==0):
			logger.info('Saving partial model to:%s',outFile)
			savethread=threading.Thread(target=nnetModel.save,kwargs={'filename':outFile});
			savethread.start()
			#nnetModel.save(filename=outFile);


		train_sets.initialize_read()

		valid_error = valid_score()
		if valid_error < best_validation_loss:
			best_validation_loss=valid_error
		logger.info('Fine Tunning:epoch %d, validation error %f',lrate.epoch, valid_error);
		lrate.get_next_rate(current_error = 100 * valid_error)

		if savethread!=None:
			savethread.join();

	end_time = time.clock()

	logger.info('Best validation error %f',best_validation_loss)

	logger.info('The Fine tunning ran for %.2fm' % ((end_time - start_time) / 60.))
	logger.info('Optimization complete with best validation score of %f %%', best_validation_loss * 100)

	return best_validation_loss

def fineTunning(nnetModel,model_config,data_spec):
	try:
		train_sets = read_dataset(data_spec['training'])
		valid_sets = read_dataset(data_spec['validation'])
	except KeyError:
		#raise e
		logger.info("No validation/training set:Skiping Fine tunning");
	else:
		try:
			outFile=model_config['output_file']
			saveFeq=model_config['save_feq']
			finetune_config = model_config['finetune_params']
			momentum = finetune_config['momentum']
			lrate = LearningRate.get_instance(finetune_config);
		except KeyError, e:
			logger.error("KeyMissing:"+str(e));
			logger.critical("Fine tunning Paramters Missing")
			exit(2)

		_fineTunning(nnetModel,train_sets,valid_sets,lrate,momentum,saveFeq,outFile)


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
	logger.info('Getting the Test(Get Label) function')
	#fp = open(out_path, "w");
	test_sets  = read_dataset(data_spec,pad_zeros=True,makeShared=False)
	# get the label function for the model
	getLabel = nnetModel.getLabelFunction()

	batch_size = test_sets.batch_size
	with open(export_path,'w') as fp:
		while (not test_sets.is_finish()):
			for batch_index in xrange(test_sets.nBatches):
				s_idx = batch_index*batch_size;
				e_idx = s_idx+batch_size;
				pred = getLabel(test_sets.feat[s_idx:e_idx])

				act = test_sets.label[s_idx:e_idx]
				if ((batch_index == test_sets.nBatches-1) and
					(not test_sets.num_pad_frames == 0)) :
						pred=pred[:-test_sets.num_pad_frames]
						act= act[:-test_sets.num_pad_frames]
				labels = zip(pred.T,act)
				numpy.savetxt(fp, labels,fmt='%d %d')
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
