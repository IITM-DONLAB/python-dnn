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


#lib imports
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

#module imports
from utils.load_conf import load_model,load_rbm_spec,load_data_spec
from models.srbm import SRBM
from io_modules.file_io import read_dataset
from io_modules import setLogger

import logging
logger = logging.getLogger(__name__)

def runRBM(configFile):
	model_config = load_model(configFile)

	rbm_config = load_rbm_spec(model_config['rbm_nnet_spec'])
	#mlp_config = load_mlp_spec(model_config['hidden_nnet_spec']);
	data_spec =  load_data_spec(model_config['data_spec']);


	#generating Random
	numpy_rng = numpy.random.RandomState(rbm_config['random_seed'])
	theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))


	srbm = SRBM(numpy_rng=numpy_rng, theano_rng = theano_rng, n_ins=rbm_config['n_ins'],
              hidden_layers_sizes=rbm_config['layers'],
              n_outs=rbm_config['n_outs'], first_layer_gb = rbm_config['first_layer_gb'])

	train_sets, train_xy, train_x, train_y = read_dataset(data_spec['training'])


	logger.info('> ... getting the pretraining functions')
	pretraining_fns = srbm.pretraining_functions(train_set_x=train_x,
                                                 batch_size=model_config['batch_size'],
                                                 weight_cost = 0.0002)

	logger.info('> ... pre-training the model')
	#start_time = time.clock()

if __name__ == '__main__':
    import sys
    setLogger();
    runRBM(sys.argv[1])
