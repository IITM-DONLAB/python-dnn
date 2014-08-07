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

import cPickle, gzip, os, time
from model.cnn import CNN;
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from utils.load_conf import load_model,load_conv_spec,load_mlp_spec,load_data_spec
from io_modules.file_io import read_dataset
#from io.file_io import read_data_args, read_dataset
from utils.learn_rates import LearningRateExpDecay,LearningRateConstant
from utils.utils import parse_activation



def runCNN(configFile):
	model_configs = load_model(configFile,'CNN')

	# learning rate
	if model_configs['l_rate_method'] =='E':
		lrate = LearningRateExpDecay(model_configs['l_rate'])
	else:
		lrate =  LearningRateConstant(model_configs['l_rate'])

	conv_configs,conv_layer_configs = load_conv_spec(model_configs['conv_nnet_spec'],model_configs['batch_size'],
				model_configs['input_shape'])
	
	mlp_configs = load_mlp_spec(model_configs['hidden_nnet_spec']);
	
	data_spec =  load_data_spec(model_configs['data_spec']);
	
	train_sets, train_xy, train_x, train_y = read_dataset(data_spec['training'])
	valid_sets, valid_xy, valid_x, valid_y = read_dataset(data_spec['validation'])

	numpy_rng = numpy.random.RandomState(89677)
	theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
	
	print('> ... building the model')
	conv_activation = parse_activation(conv_configs['activation']);
	hidden_activation = parse_activation(mlp_configs['activation']);

	cnn = CNN(numpy_rng,theano_rng,conv_layer_configs = conv_layer_configs, batch_size = model_configs['batch_size'],
		n_outs=model_configs['n_outs'],hidden_layers_sizes=mlp_configs['layers'], conv_activation = conv_activation,
		hidden_activation = hidden_activation,use_fast = conv_configs['use_fast'])


	#print('> ... getting the finetuning functions')
	#train_fn, valid_fn = cnn.build_finetune_functions((train_x, train_y),
	#		 (valid_x, valid_y), batch_size=model_configs['batch_size'])

	#start_time = time.clock()
	#while (lrate.get_rate() != 0):
	#	train_error = []
	#	while (not train_sets.is_finish()):
	#		train_sets.load_next_partition(train_xy)
	#		for batch_index in xrange(train_sets.cur_frame_num / batch_size):  # loop over mini-batches
	#			train_error.append(train_fn(index=batch_index, learning_rate = lrate.get_rate(), momentum = momentum))
	#	train_sets.initialize_read()
	#log('> epoch %d, training error %f' % (lrate.epoch, numpy.mean(train_error)))
	
	#valid_error = []
	#while (not valid_sets.is_finish()):
	#		valid_sets.load_next_partition(valid_xy)
	#		for batch_index in xrange(valid_sets.cur_frame_num / batch_size):  # loop over mini-batches
	#		valid_error.append(valid_fn(index=batch_index))
	#	valid_sets.initialize_read()
	#log('> epoch %d, lrate %f, validation error %f' % (lrate.epoch, lrate.get_rate(), numpy.mean(valid_error)))

	


if __name__ == '__main__':
	import sys
	runCNN(sys.argv[1])
