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
from io_modules import setLogger
from utils.learn_rates import LearningRate
from utils.utils import parse_activation
from io_modules.model_io import _nnet2file, _file2nnet

from run import fineTunning,testing,createDir


import logging
logger = logging.getLogger(__name__)

def preTraining(nnetModel,train_sets,train_xy,train_x,train_y,model_config):

    logger.info('Getting the pretraining functions....')
    pretraining_fns = nnetModel.pretraining_functions(train_set_x=train_x,
                                                 batch_size=model_config['batch_size'],
                                                 weight_cost = 0.0002)
    

    batch_size = model_config['batch_size'];
    pretrainingEpochs = model_config['pretraining_epochs']
    keep_layer_num=model_config['keep_layer_num']
    
    initialMomentum = model_config['initial_momentum']
    initMomentumEpochs = model_config['initial_momentum_epoch']
    finalMomentum = model_config['final_momentum']
    
    logger.info('Pre-training the model ...')
    start_time = time.clock()
    

    ## Pre-train layer-wise
    for i in range(keep_layer_num, nnetModel.nPreTrainLayers):
        if (nnetModel.rbm_layers[i].is_gbrbm()):
            pretrain_lr = model_config['gbrbm_learning_rate']
        else:
            pretrain_lr = model_config['learning_rate']
        # go through pretraining epochs
        momentum = initialMomentum
        for epoch in xrange(pretrainingEpochs):
            # go through the training set
            if (epoch > initMomentumEpochs):
                momentum = finalMomentum

            r_c, fe_c = [], []  # keep record of reconstruction and free-energy cost
            while not train_sets.is_finish():
                train_sets.make_partition_shared(train_xy)
                #train_sets.load_next_partition(train_xy)
                for batch_index in xrange(train_sets.cur_frame_num / batch_size):  
                    # loop over mini-batches
                    #logger.info("Training For epoch %d and batch %d",epoch,batch_index)
                    [reconstruction_cost, free_energy_cost] = pretraining_fns[i](index=batch_index,
                                                                             lr=pretrain_lr,
                                                                             momentum=momentum)
                    r_c.append(reconstruction_cost)
                    fe_c.append(free_energy_cost)
                train_sets.read_next_partition_data()
            train_sets.initialize_read()
            logger.info('Training layer %i, epoch %d, r_cost %f, fe_cost %f',
                i, epoch, numpy.mean(r_c), numpy.mean(fe_c))
    end_time = time.clock()
    logger.info('The PreTraing ran for %.2fm' % ((end_time - start_time) / 60.))

def runRBM(arg):

    if type(arg) is dict:
        model_config = arg
    else :
        model_config = load_model(arg,'RBM')

    rbm_config = load_rbm_spec(model_config['nnet_spec'])
    data_spec =  load_data_spec(model_config['data_spec']);


    #generating Random
    numpy_rng = numpy.random.RandomState(rbm_config['random_seed'])
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

    activationFn = parse_activation(rbm_config['activation']);
 
    createDir(model_config['wdir']);
    #create working dir

    keep_layer_num = model_config['keep_layer_num']
    batch_size = model_config['batch_size']
    wdir = model_config['wdir']
    

    dbn = DBN(numpy_rng=numpy_rng, theano_rng = theano_rng, n_ins=rbm_config['n_ins'],
            hidden_layers_sizes=rbm_config['hidden_layers'],n_outs=rbm_config['n_outs'],
            first_layer_gb = rbm_config['first_layer_gb'],
            pretrainedLayers=rbm_config['pretrained_layers'],
            activation=activationFn)


    train_sets, train_xy, train_x, train_y = read_dataset(data_spec['training'],
        model_config['batch_size'])

    if keep_layer_num > 0:
        current_nnet = wdir + '/nnet.ptr.current'
        logger.info('Initializing model from ' + str(current_nnet) + '....')
        # load model
        _file2nnet(dbn.sigmoid_layers, set_layer_num = keep_layer_num, 
            filename = current_nnet, withfinal=False)

    preTraining(dbn,train_sets,train_xy,train_x,train_y,model_config)


    ########################
    # FINETUNING THE MODEL #
    ########################

    try:
        valid_sets, valid_xy, valid_x, valid_y = read_dataset(data_spec['validation'],
            model_config['batch_size'])        
    except KeyError:
        #raise e
        logger.info("No validation set:Skiping Fine tunning");
        logger.info("Finshed")
        sys.exit(0)

    try:
        finetune_method = model_config['finetune_method']
        finetune_config = model_config['finetune_rate'] 
        momentum = model_config['finetune_momentum']
        lrate = LearningRate.get_instance(finetune_method,finetune_config);        
    except KeyError, e:
        print("KeyMissing:"+str(e));
        print("Fine tunning Paramters Missing")
        sys.exit(2)


    fineTunning(dbn,train_sets,train_xy,train_x,train_y,
        valid_sets,valid_xy,valid_x,valid_y,lrate,momentum,batch_size)


    try:
        test_sets, test_xy, test_x, test_y = read_dataset(data_spec['testing'],
            model_config['batch_size']) 
    except KeyError:
        #raise e
        logger.info("No testing set:Skiping Testing");
        logger.info("Finshed")
        sys.exit(0)

    testing(dbn,test_sets, test_xy, test_x, test_y,batch_size)

    logger.info('Saving model to ' + str(model_config['output_file']) + ' ....')
    _nnet2file(dbn.sigmoid_layers, filename=model_config['output_file'], withfinal=True)



if __name__ == '__main__':
    import sys
    setLogger();
    runRBM(sys.argv[1])
