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

import cPickle, gzip, os, sys, time, numpy, json

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from io_modules.model_io import _nnet2file, _file2nnet, _cnn2file, _file2cnn
from utils.load_conf import load_model, load_conv_spec,load_mlp_spec,load_data_spec 
from io_modules.data_exporter import export_data
from io_modules import setLogger

from models.cnn import CNN
from utils.utils import parse_activation

import logging
logger = logging.getLogger(__name__)

def runCNNFeat(arg):

	if type(arg) is dict:
		model_configs = arg
	else :
		model_configs = load_model(arg,'CNN')
	
	
	batch_size = model_configs['batch_size'];
	conv_configs,conv_layer_configs = load_conv_spec(model_configs['conv_nnet_spec'],model_configs['batch_size'],
				model_configs['input_shape'])
	mlp_configs = load_mlp_spec(model_configs['hidden_nnet_spec']);
	conv_activation = parse_activation(conv_configs['activation']);
	hidden_activation = parse_activation(mlp_configs['activation']);

	numpy_rng = numpy.random.RandomState(89677)
	theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

	logger.info('> ... Loading the model')
	cnn = CNN(numpy_rng,theano_rng,conv_layer_configs = conv_layer_configs, batch_size = batch_size,
		n_outs=model_configs['n_outs'],hidden_layers_sizes=mlp_configs['layers'], conv_activation = conv_activation,
		hidden_activation = hidden_activation,use_fast = conv_configs['use_fast'])	
	_file2cnn(cnn.conv_layers, filename=model_configs['conv_output_file'], activation=conv_activation)
	out_fn = cnn.build_out_function();
	
	logger.info('> ... Exporting the data')
	data_spec = load_data_spec(model_configs['data_spec'])
	export_data(data_spec['testing'],model_configs['export_path'],out_fn,cnn.conv_output_dim);

if __name__ == '__main__':
	setLogger(level="DEBUG");
	runCNNFeat(sys.argv[1])

