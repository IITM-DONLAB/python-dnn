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


#lib imports
import time

import numpy
import theano
from theano.tensor.shared_randomstreams import RandomStreams

#module imports
from utils.load_conf import load_model,load_rbm_spec,load_data_spec
from models.dbn import DBN
from io_modules.file_reader import read_dataset
from io_modules.data_exporter import export_data
from io_modules import setLogger
from utils.utils import parse_activation

from run import fineTunning,testing,exportFeatures
from run import createDir


import logging
logger = logging.getLogger(__name__)


def getFunction(dbn):
    (op,k) = dbn.rbm_layers[-1].propdown(dbn.features);
    in_x = dbn.x.type('in_x');
    fn = theano.function(inputs=[in_x],outputs=op,
                         givens={dbn.x: in_x},name='re')#,on_unused_input='warn')
    return fn

    

def preTraining(dbn,train_sets,pretrain_config):

    train_xy = train_sets.shared_xy
    train_x = train_sets.shared_x

    logger.info('Getting the pretraining functions....')
    batch_size = train_sets.batch_size;
    pretrainingEpochs = pretrain_config['epochs']
    keep_layer_num=pretrain_config['keep_layer_num']
    
    initialMomentum = pretrain_config['initial_momentum']
    initMomentumEpochs = pretrain_config['initial_momentum_epoch']
    finalMomentum = pretrain_config['final_momentum']

    pretraining_fns = dbn.pretraining_functions(train_set_x=train_x,
                                                 batch_size=batch_size,
                                                 weight_cost = 0.0002)
    
    logger.info('Pre-training the model ...')
    start_time = time.clock()
    
    ## Pre-train layer-wise
    for i in range(keep_layer_num, dbn.nPreTrainLayers):
        if (dbn.rbm_layers[i].is_gbrbm()):
            pretrain_lr = pretrain_config['gbrbm_learning_rate']
        else:
            pretrain_lr = pretrain_config['learning_rate']
        # go through pretraining epochs
        momentum = initialMomentum
        for epoch in xrange(pretrainingEpochs):
            # go through the training set
            if (epoch > initMomentumEpochs):
                momentum = finalMomentum

            r_c, fe_c = [], []  # keep record of reconstruction and free-energy cost
            while not train_sets.is_finish():

                for batch_index in xrange(train_sets.cur_frame_num / batch_size):  
                    # loop over mini-batches
                    #logger.info("Training For epoch %d and batch %d",epoch,batch_index)
                    [reconstruction_cost, free_energy_cost] = pretraining_fns[i](index=batch_index,
                                                                             lr=pretrain_lr,
                                                                             momentum=momentum)
                    logger.debug('Training batch %d reconstruction cost=%f,free_energy_cost=%f',
                        batch_index,reconstruction_cost,free_energy_cost);
                    
                    r_c.append(reconstruction_cost)
                    fe_c.append(free_energy_cost)
                train_sets.read_next_partition_data()

            logger.info('Training layer %i, epoch %d, reconstruction cost=%f,free_energy_cost=%f',
                i, epoch, numpy.mean(r_c), numpy.mean(fe_c))
            train_sets.initialize_read()

    end_time = time.clock()
    logger.info('The PreTraing ran for %.2fm' % ((end_time - start_time) / 60.))

def runRBM(arg):

    if type(arg) is dict:
        model_config = arg
    else :
        model_config = load_model(arg,'RBM')

    rbm_config = load_rbm_spec(model_config['nnet_spec'])
    data_spec =  load_data_spec(model_config['data_spec'],model_config['batch_size']);


    #generating Random
    numpy_rng = numpy.random.RandomState(model_config['random_seed'])
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

    activationFn = parse_activation(rbm_config['activation']);
 
    createDir(model_config['wdir']);
    #create working dir

    batch_size = model_config['batch_size']
    wdir = model_config['wdir']
    

    dbn = DBN(numpy_rng=numpy_rng, theano_rng = theano_rng, n_ins=model_config['n_ins'],
            hidden_layers_sizes=rbm_config['hidden_layers'],n_outs=model_config['n_outs'],
            first_layer_gb = rbm_config['first_layer_gb'],
            pretrainedLayers=rbm_config['pretrained_layers'],
            activation=activationFn)
    
    logger.info("Loading Pretrained network weights")
    try:
    # pretraining
        ptr_file = model_config['input_file']
        dbn.load(filename=ptr_file)
    except KeyError, e:
        logger.info("KeyMissing:"+str(e));
        logger.info("Pretrained network Missing in configFile: Skipping Loading");
    except IOError, e:
        logger.error("IOError:"+str(e));
        logger.error('Model cannot be initialize from input file ')
        sys.exit(2)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    if model_config['processes']['pretraining']:
        train_sets = read_dataset(data_spec['training'])
        preTraining(dbn,train_sets,model_config['pretrain_params'])
        del train_sets;

    ########################
    # FINETUNING THE MODEL #
    ########################
    if model_config['processes']['finetuning']:
        fineTunning(dbn,model_config,data_spec)

    ########################
    #  TESTING THE MODEL   #
    ########################
    if model_config['processes']['testing']:
        testing(dbn,data_spec)
    ##########################
    #   Export Features	##
    ##########################
    if model_config['processes']['export_data']:
	exportFeatures(dbn,model_config,data_spec)


    logger.info('Saving model to ' + str(model_config['output_file']) + '....')
    dbn.save(filename=model_config['output_file'])
    logger.info('Saved model to ' + str(model_config['output_file']))

if __name__ == '__main__':
    import sys
    setLogger();
    runRBM(sys.argv[1])
