import cPickle, gzip, os, sys, time, json

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from utils.load_conf import load_model
from io.file_io import read_data_args, read_dataset
from utils.learn_rates import LearningRateExpDecay
from utils.utils import parse_conv_spec, parse_lrate, parse_arguments


if __name__ == '__main__':
	data = load_model(sys.argv[1])
	
	# learning rate
	if data['lrate'] =='C':
		lrate = LearningRateExpDecay(start_rate=0.08,scale_by = 0.5,min_derror_decay_start = 0.05,
				 min_derror_stop = 0.05, min_epoch_decay_start=15);
	else:
		lrate = parse_lrate(arguments['lrate']) ;
	
	# batch_size and momentum
	batch_size=256; momentum=0.5;
	if arguments.has_key('batch_size'):
		batch_size = int(arguments['batch_size'])
	if arguments.has_key('momentum'):
		momentum = float(arguments['momentum'])
	
	# conv layer configuraitons
	conv_layer_configs = parse_conv_spec(conv_nnet_spec, batch_size)


	# full layer configurations
	nnet_layers = full_nnet_spec.split(":")
	hidden_layers_sizes = []
	for i in range(0, len(nnet_layers)-1):
		hidden_layers_sizes.append(int(nnet_layers[i]))
	n_outs = int(nnet_layers[-1])

	conv_activation = T.nnet.sigmoid
	full_activation = T.nnet.sigmoid
	if arguments.has_key('conv_activation'):
		conv_activation = parse_activation(arguments['conv_activation'])
	if arguments.has_key('full_activation'):
		full_activation = parse_activation(arguments['full_activation'])

	# whether to use the fast version of CNN with pylearn2
	use_fast = False
	if arguments.has_key('use_fast'):
		use_fast = string_2_bool(arguments['use_fast'])	
####################################################################################################################################################
	# mandatory arguments
	train_data_spec = arguments['train_data']
	valid_data_spec = arguments['valid_data']
	conv_nnet_spec = arguments['conv_nnet_spec']
	full_nnet_spec = arguments['full_nnet_spec']
	wdir = arguments['wdir']
	conv_output_file = arguments['conv_output_file']
	full_output_file = arguments['full_output_file']


from io_func.model_io import _nnet2file, _file2nnet, _cnn2file, _file2cnn, log


from utils.utils import parse_conv_spec, parse_lrate, parse_arguments, parse_activation, activation_to_txt, string_2_bool

from models.cnn import CNN


	
	
	if arguments.has_key('conv_ptr_file'):
		if not arguments.has_key('conv_ptr_layer_number'):
			print "Error: --conv-ptr-layer-number should be provided together with --conv-ptr-file"
			exit(1)
		conv_ptr_file = arguments['conv_ptr_file']
		conv_ptr_layer_number = int(arguments['conv_ptr_layer_number'])
	full_ptr_layer_number = 0
	if arguments.has_key('full_ptr_file'):
		if not arguments.has_key('full_ptr_layer_number'):
			print "Error: --full-ptr-layer-number should be provided together with --full-ptr-file"
			exit(1)
		full_ptr_file = arguments['full_ptr_file']
		full_ptr_layer_number = int(arguments['full_ptr_layer_number'])

	# the index of layers which will be updated; it's possible that we may want to update only some of
	# the layers, instead of all of them
	update_layers = [i for i in xrange(50)]
	if arguments.has_key('update_layers'):
		layers = arguments['update_layers'].split(':')
		update_layers = [int(layer) for layer in layers] 

	# output format: kaldi or janus
	output_format = 'kaldi'
	if arguments.has_key('output_format'):
		output_format = arguments['output_format']
	if output_format != 'kaldi':
		print "Error: the output format only supports Kaldi"
		exit(1)
 
	train_dataset, train_dataset_args = read_data_args(train_data_spec)
	valid_dataset, valid_dataset_args = read_data_args(valid_data_spec)
	
	# reading data 
	train_sets, train_xy, train_x, train_y = read_dataset(train_dataset, train_dataset_args)
	valid_sets, valid_xy, valid_x, valid_y = read_dataset(valid_dataset, valid_dataset_args)

	numpy_rng = numpy.random.RandomState(89677)
	theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
	log('> ... building the model')
	# construct the cnn architecture
	cnn = CNN(numpy_rng=numpy_rng, theano_rng = theano_rng,
			  batch_size = batch_size, n_outs=n_outs,
			  conv_layer_configs = conv_layer_configs,
			  hidden_layers_sizes = hidden_layers_sizes,
			  conv_activlogation = conv_activation, 
			  full_activation = full_activation,
			  use_fast = use_fast, update_layers = update_layers)

####################################################################################################################################################
	total_layer_number = len(cnn.layers)
	if full_ptr_layer_number > 0:
		_file2nnet(cnn.layers[len(conv_layer_configs):total_layer_number], set_layer_num = full_ptr_layer_number, filename = full_ptr_file,  withfinal=False)
	if conv_ptr_layer_number > 0:
		_file2cnn(cnn.layers[0:len(conv_layer_configs)], filename=conv_ptr_file)
	# get the training, validation and testing function for the model
	log('> ... getting the finetuning functions')
	train_fn, valid_fn = cnn.build_finetune_functions(
				(train_x, train_y), (valid_x, valid_y),
				batch_size=batch_size)

	log('> ... finetunning the model')
	start_time = time.clock()

	while (lrate.get_rate() != 0):
		train_error = []
		while (not train_sets.is_finish()):
			train_sets.load_next_partition(train_xy)
			for batch_index in xrange(train_sets.cur_frame_num / batch_size):  # loop over mini-batches
			train_error.append(train_fn(index=batch_index, learning_rate = lrate.get_rate(), momentum = momentum))
		train_sets.initialize_read()
	log('> epoch %d, training error %f' % (lrate.epoch, numpy.mean(train_error)))

	valid_error = []
	while (not valid_sets.is_finish()):
			valid_sets.load_next_partition(valid_xy)
			for batch_index in xrange(valid_sets.cur_frame_num / batch_size):  # loop over mini-batches
			valid_error.append(valid_fn(index=batch_index))
		valid_sets.initialize_read()
	log('> epoch %d, lrate %f, validation error %f' % (lrate.epoch, lrate.get_rate(), numpy.mean(valid_error)))

	lrate.get_next_rate(current_error = 100 * numpy.mean(valid_error))

	# output conv layer config
	for i in xrange(len(conv_layer_configs)):
		conv_layer_configs[i]['activation'] = activation_to_txt(conv_activation)
		with open(wdir + '/conv.config.' + str(i), 'wb') as fp:
			json.dump(conv_layer_configs[i], fp, indent=2, sort_keys = True)
			fp.flush()

	# output the conv part
	_cnn2file(cnn.layers[0:len(conv_layer_configs)], filename=conv_output_file)
	# output the full part
	_nnet2file(cnn.layers[len(conv_layer_configs):total_layer_number], filename=wdir + '/nnet.finetune.tmp')
	_nnet2kaldi(str(cnn.conv_output_dim) + ':' + full_nnet_spec, filein = wdir + '/nnet.finetune.tmp', fileout = full_output_file)

	end_time = time.clock()
	print >> sys.stderr, ('The code for file ' +
						  os.path.split(__file__)[1] +
						  ' ran for %.2fm' % ((end_time - start_time) / 60.))

