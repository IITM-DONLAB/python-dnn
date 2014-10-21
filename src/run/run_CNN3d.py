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
from models.cnn3d import CNN3D
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from utils.load_conf import load_model,load_conv_spec,load_data_spec,__debugPrintData__
from io_modules.file_reader import read_dataset
from utils.utils import parse_activation
from io_modules import setLogger

from run import fineTunning,testing,exportFeatures,createDir

import logging
logger = logging.getLogger(__name__)


def runCNN3D(arg):
	
	if type(arg) is dict:
		model_config = arg
	else :
		model_config = load_model(arg,'CNN')
	
	conv_config,conv_layer_config,mlp_config = load_conv_spec(
			model_config['nnet_spec'],
			model_config['batch_size'],
			model_config['input_shape'])
	#__debugPrintData__(conv_layer_config,'covolution config')
	
	data_spec =  load_data_spec(model_config['data_spec'],model_config['batch_size']);
	
	numpy_rng = numpy.random.RandomState(model_config['random_seed'])
	theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
	
	logger.info('> ... building the model')
	conv_activation = parse_activation(conv_config['activation']);
	hidden_activation = parse_activation(mlp_config['activation']);

	createDir(model_config['wdir']);
	
	#create working dir

	batch_size = model_config['batch_size'];
	
	cnn = CNN3D(numpy_rng,theano_rng,conv_layer_configs = conv_layer_config, batch_size = batch_size,
			n_outs=model_config['n_outs'],hidden_layer_configs=mlp_config, 
			conv_activation = conv_activation,hidden_activation = hidden_activation,
			l1_reg = mlp_config['l1_reg'],l2_reg = mlp_config['l1_reg'],max_col_norm = mlp_config['max_col_norm'])
	
	"""			
	########################
	 # Loading  THE MODEL #
	########################
	try:
		# pretraining
		ptr_file = model_config['input_file']
		pretrained_layers = mlp_config['pretrained_layers']
		logger.info("Loading the pretrained network..")
		cnn.load(filename=ptr_file,max_layer_num = pretrained_layers,  withfinal=True)
	except KeyError, e:
		logger.warning("Pretrained network missing in working directory, skipping model loading")
	except IOError, e:
		logger.error("IOError:"+str(e));
		logger.error('Model cannot be initialize from input file ')
		exit(2)

	"""	
	
	
	########################
	# FINETUNING THE MODEL #
	########################
	if model_config['processes']['finetuning']:
		fineTunning(cnn,model_config,data_spec)
		
	
	########################
	#  TESTING THE MODEL   #
	########################
	if model_config['processes']['testing']:
		testing(cnn,data_spec)
	
	"""
	##########################
	##   Export Features    ##
	##########################
	if model_config['processes']['export_data']:
		exportFeatures(cnn,model_config,data_spec)

	logger.info('Saving model to ' + str(model_config['output_file'])+ '....')
	cnn.save(filename=model_config['output_file']);
	logger.info('Saved model to ' + str(model_config['output_file']))


	##############################
	##	Plotting  Layer output ##
	##############################
	if model_config['processes']['plotting']:
		cnn.plot_layer_output(data_spec['validation'],model_config['plot_path']);
	"""
		
if __name__ == '__main__':
	runCNN(sys.argv[1])
