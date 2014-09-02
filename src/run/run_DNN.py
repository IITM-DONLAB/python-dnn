#!/usr/bin/env python2.7
# Copyright 2014    G.K SUDHARSHAN <sudharpun90@gmail.comIIT Madras
# Copyright 2014    Abil N George<mail@abilng.inIIT Madras
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
from utils.learn_rates import LearningRate
from utils.utils import parse_activation
from io_modules.model_io import _nnet2file, _file2nnet

from run import fineTunning,testing,exportFeatures,createDir

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
    numpy_rng = numpy.random.RandomState(dnn_config['random_seed'])
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

    activationFn = parse_activation(dnn_config['activation']);

    #create working dir
    createDir(model_config['wdir']);

    batch_size = model_config['batch_size'];

    max_col_norm = dnn_config['max_col_norm']
    l1_reg = dnn_config['l1_reg']
    l2_reg = dnn_config['l2_reg']

    n_ins = model_config['n_ins']
    n_outs = model_config['n_outs']

    do_maxout = dnn_config['do_maxout']
    pool_size = dnn_config['pool_size']
    do_dropout = dnn_config['do_dropout']
    hidden_layers_sizes = dnn_config['hidden_layers']

    logger.info('Building the model')

    if do_dropout:
        dropout_factor = dnn_config['dropout_factor']
        input_dropout_factor = dnn_config['input_dropout_factor']

        dnn = DNN_Dropout(numpy_rng=numpy_rng, theano_rng = theano_rng, n_ins=n_ins,
              hidden_layers_sizes=hidden_layers_sizes, n_outs=n_outs,
              activation = activationFn, dropout_factor = dropout_factor,
              input_dropout_factor = input_dropout_factor,
              do_maxout = do_maxout, pool_size = pool_size,
              max_col_norm = max_col_norm, l1_reg = l1_reg, l2_reg = l2_reg)
    else:
        do_pnorm = dnn_config['do_pnorm']
        pnorm_order = dnn_config['pnorm_order']
        
        dnn = DNN(numpy_rng=numpy_rng, theano_rng = theano_rng, n_ins=n_ins,
              hidden_layers_sizes=hidden_layers_sizes, n_outs=n_outs,
              activation = activationFn, do_maxout = do_maxout, pool_size = pool_size,
              do_pnorm = do_pnorm, pnorm_order = pnorm_order,
              max_col_norm = max_col_norm, l1_reg = l1_reg, l2_reg = l2_reg)


    try:
        # pretraining
        ptr_file = model_config['input_file']
        pretrained_layers = dnn_config['pretrained_layers']

        _file2nnet(dnn.mlp_layers, set_layer_num = pretrained_layers,
            filename = ptr_file,  withfinal=False)
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
        try:
            train_sets, train_xy, train_x, train_y = read_dataset(data_spec['training'])
            valid_sets, valid_xy, valid_x, valid_y = read_dataset(data_spec['validation'])
        except KeyError:
            #raise e
            logger.error("No validation/Test set:Skiping Fine tunning");
        else:
            try:
                finetune_method = model_config['finetune_method']
                finetune_config = model_config['finetune_rate']
                momentum = model_config['finetune_momentum']
                lrate = LearningRate.get_instance(finetune_method,finetune_config);
            except KeyError, e:
                logger.error("KeyMissing:"+str(e));
                logger.error("Fine tunning Paramters Missing")
                sys.exit(2)


            fineTunning(dnn,train_sets,train_xy,train_x,train_y,
                valid_sets,valid_xy,valid_x,valid_y,lrate,momentum,batch_size)


    ########################
    #  TESTING THE MODEL   #
    ########################
    if model_config['processes']['testing']:
        try:
            test_sets, test_xy, test_x, test_y = read_dataset(data_spec['testing'])
        except KeyError:
            #raise e
            logger.info("No testing set:Skiping Testing");
        else:
            testing(dnn,test_sets, test_xy, test_x, test_y,batch_size)

    ##########################
    ##   Export Features    ##
    ##########################
    if model_config['processes']['export_data']:
        try:
            exportFeatures(dnn,model_config['export_path'],data_spec['testing'])
        except KeyError:
            #raise e
            logger.info("No testing set:Skiping Exporting");


    logger.info('Saving model to ' + str(model_config['output_file']) + '....')

    if do_dropout:
        _nnet2file(dnn.mlp_layers, filename=model_config['output_file'],
        input_factor = input_dropout_factor, factor = dropout_factor)
    else:
        _nnet2file(dnn.mlp_layers, filename=model_config['output_file'])

if __name__ == '__main__':
    import sys
    setLogger(level="INFO");
    logger.info('Stating....');
    runDNN(sys.argv[1]);
    sys.exit(0)