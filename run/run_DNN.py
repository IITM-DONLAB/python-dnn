#!/usr/bin/env python2.7
# Copyright 2014	G.K SUDHARSHAN <sudharpun90@gmail.comIIT Madras
# Copyright 2014	Abil N George<mail@abilng.inIIT Madras
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

import time,sys
import numpy
import theano

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from models.dnn import DNN
from models.dropout_nnet import DNN_Dropout
from utils.load_conf import load_model,load_sda_spec,load_data_spec
from io_modules.file_reader import read_dataset
from io_modules import setLogger
from utils.learn_rates import LearningRate

from io_modules.model_io import _nnet2file, _file2nnet
from models import fineTunning,testing

import logging
logger = logging.getLogger(__name__)


def runDNN(configFile):

	model_config = load_model(configFile)
	dnn_config = load_dnn_spec(model_config['dnn_nnet_spec'])
	data_spec =  load_data_spec(model_config['data_spec']);


	#generating Random
	numpy_rng = numpy.random.RandomState(dnn_config['random_seed'])
	theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

	# pretraining
	ptr_file = dnn_config['ptr_file']
	ptr_layer_number = dnn_config['ptr_layer_number'])

	max_col_norm = dnn_config['max_col_norm']
	l1_reg = dnn_config['l1_reg']
	l2_reg = dnn_config['l2_reg']


	# learning rate
	lrate = LearningRate.get_instance(model_configs['l_rate_method'],
		model_configs['l_rate']);

	# batch_size and momentum
	batch_size = model_configs['batch_size'];
	momentum = model_configs['momentum']


	n_ins = dnn_configs['n_ins']
	hidden_layers_sizes = dnn_config['hidden_layers']
	n_outs = dnn_configs['n_outs']
    
    if dnn_configs['activation'] == 'sigmoid':
        activation = T.nnet.sigmoid
    else:
        activation = T.tanh

    do_maxout = dnn_configs['do_maxout']
    pool_size = dnn_configs['pool_size']
    do_pnorm = dnn_configs['do_pnorm']
    pnorm_order = dnn_configs['pnorm_order']

    do_dropout = dnn_configs['do_dropout']
    dropout_factor = dnn_configs['dropout_factor']
    input_dropout_factor = dnn_configs['input_dropout_factor']

    train_sets, train_xy, train_x, train_y = read_dataset(data_spec['training'])
    valid_sets, valid_xy, valid_x, valid_y = read_dataset(data_spec['validation'])


	numpy_rng = numpy.random.RandomState(89677)
	theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
	
    logger.info('Building the model')
	if do_dropout:
		dnn = DNN_Dropout(numpy_rng=numpy_rng, theano_rng = theano_rng, n_ins=n_ins,
			  hidden_layers_sizes=hidden_layers_sizes, n_outs=n_outs,
			  activation = activation, dropout_factor = dropout_factor, input_dropout_factor = input_dropout_factor,
			  do_maxout = do_maxout, pool_size = pool_size,
			  max_col_norm = max_col_norm, l1_reg = l1_reg, l2_reg = l2_reg)
	else:
		dnn = DNN(numpy_rng=numpy_rng, theano_rng = theano_rng, n_ins=n_ins,
			  hidden_layers_sizes=hidden_layers_sizes, n_outs=n_outs,
			  activation = activation, do_maxout = do_maxout, pool_size = pool_size,
			  do_pnorm = do_pnorm, pnorm_order = pnorm_order,
			  max_col_norm = max_col_norm, l1_reg = l1_reg, l2_reg = l2_reg)

	if ptr_layer_number > 0:
	  _file2nnet(dnn.sigmoid_layers, set_layer_num = ptr_layer_number, filename = ptr_file,  withfinal=False)

	# get the training, validation and testing function for the model
	logger.info('Getting the finetuning functions')
	train_fn, valid_fn = dnn.build_finetune_functions(
				(train_x, train_y), (valid_x, valid_y),
				batch_size=batch_size)

	logger.info('Finetunning the model')
	start_time = time.clock()
	while (lrate.get_rate() != 0):
		train_error = []
		while (not train_sets.is_finish()):
			train_sets.load_next_partition(train_xy)
			for batch_index in xrange(train_sets.cur_frame_num / batch_size):  # loop over mini-batches
				train_error.append(train_fn(index=batch_index, learning_rate = lrate.get_rate(), momentum = momentum))
		train_sets.initialize_read()
		logger.info('Epoch %d, training error %f' % (lrate.epoch, numpy.mean(train_error)))

		valid_error = []
		while (not valid_sets.is_finish()):
			valid_sets.load_next_partition(valid_xy)
			for batch_index in xrange(valid_sets.cur_frame_num / batch_size):  # loop over mini-batches
				valid_error.append(valid_fn(index=batch_index))
		valid_sets.initialize_read()
		logger.info('Epoch %d, lrate %f, validation error %f' % (lrate.epoch, lrate.get_rate(), numpy.mean(valid_error)))

		lrate.get_next_rate(current_error = 100 * numpy.mean(valid_error))

	if do_dropout:
		_nnet2file(dnn.sigmoid_layers, filename=wdir + '/nnet.finetune.tmp', input_factor = input_dropout_factor, factor = dropout_factor)
	else:
		_nnet2file(dnn.sigmoid_layers, filename=wdir + '/nnet.finetune.tmp')

	# determine whether it's BNF based on layer sizes
	set_layer_num = -1
	withfinal = True
	bnf_layer_index = 1
	while bnf_layer_index < len(hidden_layers_sizes):
		if hidden_layers_sizes[bnf_layer_index] < hidden_layers_sizes[bnf_layer_index - 1]:  
			break
		bnf_layer_index = bnf_layer_index + 1

	if bnf_layer_index < len(hidden_layers_sizes):  # is bottleneck
		set_layer_num = bnf_layer_index+1
		withfinal = False

	end_time = time.clock()

    logger.info('The Training ran for %.2fm' % ((end_time - start_time) / 60.))

    if do_maxout:
        _nnet2janus_maxout(nnet_spec, pool_size = pool_size, set_layer_num = set_layer_num, filein = wdir + '/nnet.finetune.tmp', fileout = output_file, withfinal=withfinal)
    else:
        _nnet2janus(nnet_spec, set_layer_num = set_layer_num, filein = wdir + '/nnet.finetune.tmp', fileout = output_file, withfinal=withfinal)
    



if __name__ == '__main__':
	import sys
    setLogger(level="INFO");
    logger.info('Stating....');
    runDNN(sys.argv[1]);
    sys.exit(0)
