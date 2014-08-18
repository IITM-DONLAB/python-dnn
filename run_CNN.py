#!/usr/bin/env python2.7
# Copyright 2014	G.K SUDHARSHAN <sudharpun90@gmail.com>    IIT Madras
# Copyright 2014	Abil N George<mail@abilng.in>    IIT Madras
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

import cPickle, gzip, os, time,sys
from models.cnn import CNN;
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from utils.load_conf import load_model,load_conv_spec,load_mlp_spec,load_data_spec
from io_modules.file_reader import read_dataset
from utils.learn_rates import LearningRate
from utils.utils import parse_activation
from io_modules.model_io import _cnn2file,_nnet2file
from io_modules import setLogger

import logging
logger = logging.getLogger(__name__)


def runCNN(configFile):
	model_configs = load_model(configFile,'CNN')

	# learning rate, batch-size and momentum
	lrate = LearningRate.get_instance(model_configs['l_rate_method'],model_configs['l_rate']);
	batch_size = model_configs['batch_size'];
	momentum = model_configs['momentum']


	conv_configs,conv_layer_configs = load_conv_spec(model_configs['conv_nnet_spec'],model_configs['batch_size'],
				model_configs['input_shape'])
	
	mlp_configs = load_mlp_spec(model_configs['hidden_nnet_spec']);
	
	data_spec =  load_data_spec(model_configs['data_spec']);
	
	train_sets, train_xy, train_x, train_y = read_dataset(data_spec['training'])
	valid_sets, valid_xy, valid_x, valid_y = read_dataset(data_spec['validation'])

	numpy_rng = numpy.random.RandomState(89677)
	theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
	
	logger.info('> ... building the model')
	conv_activation = parse_activation(conv_configs['activation']);
	hidden_activation = parse_activation(mlp_configs['activation']);
	
	cnn = CNN(numpy_rng,theano_rng,conv_layer_configs = conv_layer_configs, batch_size = batch_size,
		n_outs=model_configs['n_outs'],hidden_layers_sizes=mlp_configs['layers'], conv_activation = conv_activation,
		hidden_activation = hidden_activation,use_fast = conv_configs['use_fast'])

	logger.info('> ... getting the finetuning functions')
	train_fn, valid_fn = cnn.build_finetune_functions((train_x, train_y),
			 (valid_x, valid_y), batch_size=model_configs['batch_size'])

	start_time = time.clock()
	while (lrate.get_rate() != 0):
		train_error = []
		while not train_sets.is_finish():
			train_sets.make_partition_shared(train_xy)
			for batch_index in xrange(train_sets.cur_frame_num / batch_size):  # loop over mini-batches
				train_error.append(train_fn(index=batch_index, learning_rate = lrate.get_rate(), momentum = momentum))
				logger.debug('>>>> training batch %d error %f' % (batch_index, numpy.mean(train_error)))
			train_sets.read_next_partition_data()
		logger.info('> epoch %d, training error %f' % (lrate.epoch, numpy.mean(train_error)))
		train_sets.initialize_read()
	
	
		valid_error = []
		while (not valid_sets.is_finish()):
			valid_sets.make_partition_shared(valid_xy)
			for batch_index in xrange(valid_sets.cur_frame_num / batch_size):  # loop over mini-batches
				valid_error.append(valid_fn(index=batch_index))
				logger.debug('>>>> validation batch %d error %f' % (batch_index, numpy.mean(train_error)))
			valid_sets.read_next_partition_data()
		logger.info('> epoch %d, lrate %f, validation error %f' % (lrate.epoch, lrate.get_rate(), numpy.mean(valid_error)))
		valid_sets.initialize_read()
		lrate.get_next_rate(current_error = 100 * numpy.mean(valid_error))

	
	_cnn2file(cnn.layers[0:cnn.conv_layer_num], filename=model_configs['conv_output_file'],activation=conv_configs['activation']);
	_nnet2file(cnn.layers[cnn.conv_layer_num:], filename=model_configs['hidden_output_file'],activation=mlp_configs['activation']);
	
if __name__ == '__main__':
	setLogger();
	runCNN(sys.argv[1])
