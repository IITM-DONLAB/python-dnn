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

from utils.load_conf import load_model,load_dnn_spec,load_data_spec
from io_modules.file_reader import read_dataset
from io_modules import setLogger
from utils.utils import parse_activation

from run import fineTunning,testing,exportFeatures
from run import createDir

from models.dnn import DNN
from models.dropout_nnet import DNN_Dropout

import logging
logger = logging.getLogger(__name__)


def runDNN(arg):

	if type(arg) is dict:
		model_config = arg
	else :
		model_config = load_model(arg,'DNN')

	dnn_config = load_dnn_spec(model_config['nnet_spec'])
	data_spec =  load_data_spec(model_config['data_spec'],model_config['batch_size']);


	#generating Random
	numpy_rng = numpy.random.RandomState(89677)
	theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
	
	activationFn = parse_activation(dnn_config['activation']);

	#create working dir
	createDir(model_config['wdir']);
	
	batch_size = model_config['batch_size'];
	n_ins = model_config['n_ins']
	n_outs = model_config['n_outs']
	
	max_col_norm = dnn_config['max_col_norm']
	l1_reg = dnn_config['l1_reg']
	l2_reg = dnn_config['l2_reg']	
	adv_activation = dnn_config['adv_activation']
	hidden_layers_sizes = dnn_config['hidden_layers']
	do_dropout = dnn_config['do_dropout']
	logger.info('Building the model')

	if do_dropout:
		dropout_factor = dnn_config['dropout_factor']
		input_dropout_factor = dnn_config['input_dropout_factor']

		dnn = DNN_Dropout(numpy_rng=numpy_rng, theano_rng = theano_rng, n_ins=n_ins,
			  hidden_layers_sizes=hidden_layers_sizes, n_outs=n_outs,
			  activation = activationFn, dropout_factor = dropout_factor,
			  input_dropout_factor = input_dropout_factor, adv_activation = adv_activation,
			  max_col_norm = max_col_norm, l1_reg = l1_reg, l2_reg = l2_reg)
	else:
		
		dnn = DNN(numpy_rng=numpy_rng, theano_rng = theano_rng, n_ins=n_ins,
			  hidden_layers_sizes=hidden_layers_sizes, n_outs=n_outs,
			  activation = activationFn, adv_activation = adv_activation,
			  max_col_norm = max_col_norm, l1_reg = l1_reg, l2_reg = l2_reg)


	logger.info("Loading Pretrained network weights")
	try:
		# pretraining
		ptr_file = model_config['input_file']
		pretrained_layers = dnn_config['pretrained_layers']
		dnn.load(filename=ptr_file,max_layer_num = pretrained_layers,  withfinal=True)
	except KeyError, e:
		logger.critical("KeyMissing:"+str(e));
		logger.error("Pretrained network Missing in configFile")
		sys.exit(2)
	except IOError, e:
		logger.error("IOError:"+str(e));
		logger.error('Model cannot be initialize from input file ')
		sys.exit(2)

	########################
	# FINETUNING THE MODEL #
	########################
	if model_config['processes']['finetuning']:
		fineTunning(dnn,model_config,data_spec)

	########################
	#  TESTING THE MODEL   #
	########################
	if model_config['processes']['testing']:
		testing(dnn,data_spec)

	##########################
	##   Export Features	##
	##########################
	if model_config['processes']['export_data']:
		exportFeatures(dnn,model_config,data_spec)


	logger.info('Saving model to ' + str(model_config['output_file']) + '....')
	dnn.save(filename=model_config['output_file'])
	logger.info('Saved model to ' + str(model_config['output_file']))

if __name__ == '__main__':
	import sys
	setLogger(level="INFO");
	logger.info('Stating....');
	runDNN(sys.argv[1]);
	sys.exit(0)
