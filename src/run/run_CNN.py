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
from utils.utils import parse_activation
from io_modules import setLogger

from run import fineTunning,testing,exportFeatures,createDir

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

	data_spec =  load_data_spec(model_config['data_spec'],model_config['batch_size']);

	
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

	########################
	# FINETUNING THE MODEL #
	########################
	if model_config['processes']['finetuning']:
		fineTunning(cnn,model_config,data_spec)

	########################
	#  TESTING THE MODEL   #
	########################
	if model_config['processes']['testing']:
		testing(cnn,model_config,data_spec)

	##########################
	##   Export Features    ##
	##########################
	if model_config['processes']['export_data']:
		exportFeatures(cnn,model_config,data_spec)


	cnn.load(filename=model_config['output_file']);

	
if __name__ == '__main__':
	runCNN(sys.argv[1])
