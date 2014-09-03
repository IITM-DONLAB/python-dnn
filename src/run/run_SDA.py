#!/usr/bin/env python2.7
# Copyright 2014    G.K SUDHARSHAN <sudharpun90@gmail.com> IIT Madras
# Copyright 2014    Abil N George<mail@abilng.in> IIT Madras
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
import time,sys
import numpy
import theano

#module imports
from io_modules.file_reader import read_dataset
from io_modules import setLogger
from utils.utils import parse_activation
from utils.load_conf import load_model,load_sda_spec,load_data_spec

from models.sda import SDA
from run import fineTunning,testing,exportFeatures,createDir


import logging
logger = logging.getLogger(__name__)


def preTraining(layers,epochs,pretrainfns,train_sets,train_xy,corruptions,lr,batch_size):
    ## Pre-train layer-wise
    for i in xrange(layers):
         # go through pretraining epochs
        for epoch in xrange(epochs):
            # go through the training set
            c = []  # keep record of cost
            while not train_sets.is_finish():
                train_sets.make_partition_shared(train_xy)
                
                for batch_index in xrange(train_sets.cur_frame_num / batch_size):  
                    # loop over mini-batches
                    logger.debug("Training For epoch %d and batch %d",epoch,batch_index)
                    curcost = pretrainfns[i](index=batch_index,
                        corruption=corruptions[i],lr=lr)
                    c.append(curcost)
                train_sets.read_next_partition_data()
            train_sets.initialize_read()
            err = numpy.mean(c);
            logger.info("Pre-training layer %i, epoch %d, cost %f", i, epoch,err)
    return err


def runSdA(arg):

    if type(arg) is dict:
        model_config = arg
    else :
        model_config = load_model(arg,'SDA')
        
    sda_config = load_sda_spec(model_config['nnet_spec'])
    data_spec =  load_data_spec(model_config['data_spec'],model_config['batch_size']);

    train_sets, train_xy, train_x, train_y = read_dataset(data_spec['training'])

    # numpy random generator
    numpy_rng = numpy.random.RandomState(sda_config['random_seed'])
    #theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

    #get Activation function
    activationFn = parse_activation(sda_config['activation']);

    createDir(model_config['wdir']);
    #create working dir

    logger.info('building the model')
    # construct the stacked denoising autoencoder class
    sda = SDA(numpy_rng=numpy_rng, n_ins=model_config['n_ins'],
              hidden_layers_sizes=sda_config['hidden_layers'],
              n_outs=model_config['n_outs'],activation=activationFn)

    batch_size = model_config['batch_size'];


    #########################
    # PRETRAINING THE MODEL #
    #########################
    if model_config['processes']['pretraining']:
        
        train_sets, train_xy, train_x, train_y = read_dataset(data_spec['training'])
        
        pretraining_epochs= model_config['pretraining_epochs']
        pretrain_lr = model_config['pretrain_lr']

        corruption_levels =sda_config['corruption_levels']

        logger.info('Getting the pretraining functions....')
        pretraining_fns = sda.pretraining_functions(train_x=train_x,
                                                batch_size=batch_size)

        logger.info('Pre-training the model ...')
    
        start_time = time.clock()
        err=preTraining(sda.n_layers,pretraining_epochs,pretraining_fns,
            train_sets,train_xy,corruption_levels,pretrain_lr,batch_size);
        end_time = time.clock()

        logger.info('The PreTraing ran for %.2fm' % ((end_time - start_time) / 60.))


    ########################
    # FINETUNING THE MODEL #
    ########################
    if model_config['processes']['finetuning']:
        fineTunning(sda,model_config,data_spec)

    ########################
    #  TESTING THE MODEL   #
    ########################
    if model_config['processes']['testing']:
        testing(sda,model_config,data_spec)

    ##########################
    ##   Export Features    ##
    ##########################
    if model_config['processes']['export_data']:
        exportFeatures(sda,model_config,data_spec)

    # save the pretrained nnet to file
    logger.info('Saving model to ' + str(model_config['output_file']) + '....')
    sda.save(filename=model_config['output_file'], withfinal=True)


    #print test_pred


if __name__ == '__main__':
    setLogger(level="INFO");
    logger.info('Stating....');
    runSdA(sys.argv[1]);
    sys.exit(0)
