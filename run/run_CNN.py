#!/usr/bin/env python2.7
# Copyright 2014	G.K SUDHARSHAN <sudharpun90@gmail.com>	IIT Madras
# Copyright 2014	Abil N George<mail@abilng.in>	IIT Madras
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

from utils.load_conf import load_model,load_conv_spec,load_data_spec
from io_modules.file_reader import read_dataset
from utils.learn_rates import LearningRate
from utils.utils import parse_activation
from io_modules.model_io import _cnn2file,_nnet2file
from io_modules import setLogger

from run import fineTunning,testing,createDir

import logging
logger = logging.getLogger(__name__)


def runCNN(arg):
	
	if type(arg) is dict:
		model_config = arg
	else :
		model_config = load_model(arg,'CNN')
	
	conv_config,conv_layer_config,mlp_config = load_conv_spec(
			model_config['nnet_spec'],
			model_config['batch_size'],
			model_config['input_shape'])

	data_spec =  load_data_spec(model_config['data_spec']);

	
	numpy_rng = numpy.random.RandomState(89677)
	theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
	
	logger.info('> ... building the model')
	conv_activation = parse_activation(conv_config['activation']);
	hidden_activation = parse_activation(mlp_config['activation']);

	createDir(model_config['wdir']);
	#create working dir

	batch_size = model_config['batch_size'];
	cnn = CNN(numpy_rng,theano_rng,conv_layer_configs = conv_layer_config, batch_size = batch_size,
			n_outs=model_config['n_outs'],hidden_layers_sizes=mlp_config['layers'], 
			conv_activation = conv_activation,hidden_activation = hidden_activation,
			use_fast = conv_config['use_fast'])

	if model_config['processes']['finetuning']:
		
		#learning rate, batch-size and momentum
		lrate = LearningRate.get_instance(model_config['l_rate_method'],model_config['l_rate']);
		momentum = model_config['momentum']

		train_sets, train_xy, train_x, train_y = read_dataset(data_spec['training'],model_config['batch_size'])
		valid_sets, valid_xy, valid_x, valid_y = read_dataset(data_spec['validation'],model_config['batch_size'])

		err=fineTunning(cnn,train_sets,train_xy,train_x,train_y,
			valid_sets,valid_xy,valid_x,valid_y,lrate,momentum,batch_size);

	####################
	##	TESTING	 ##
	####################
	if model_config['processes']['testing']:
		try:
			test_sets, test_xy, test_x, test_y = read_dataset(data_spec['testing'],model_config['batch_size']) 
		except KeyError:
			#raise e
			logger.info("No testing set:Skiping Testing");
			logger.info("Finshed")
			sys.exit(0)

		pred,err=testing(cnn,test_sets, test_xy, test_x, test_y,batch_size)

	_cnn2file(cnn.layers[0:cnn.conv_layer_num], filename=model_config['conv_output_file'],activation=conv_config['activation']);
	_nnet2file(cnn.layers[cnn.conv_layer_num:], filename=model_config['hidden_output_file'],activation=mlp_config['activation']);
	
if __name__ == '__main__':
	setLogger(level="DEBUG");
	runCNN(sys.argv[1])
