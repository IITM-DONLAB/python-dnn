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

#module imports
from utils.load_conf import load_model,load_sda_spec,load_data_spec
from io_modules.file_io import read_dataset
from io_modules import setLogger
from models.sda import SDA


import logging
logger = logging.getLogger(__name__)

def runSdA(configFile):

    model_config = load_model(configFile)
    sda_config = load_sda_spec(model_config['sda_nnet_spec'])
    data_spec =  load_data_spec(model_config['data_spec']);


    train_sets, train_xy, train_x, train_y = read_dataset(data_spec['training'])

    # numpy random generator
    numpy_rng = numpy.random.RandomState(sda_config['random_seed'])
    #theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

    logger.info('building the model')
    # construct the stacked denoising autoencoder class
    sda = SDA(numpy_rng=numpy_rng, n_ins=sda_config['n_ins'],
              hidden_layers_sizes=sda_config['hidden_layers'],
              n_outs=sda_config['n_outs'])



    batch_size = model_config['batch_size'];
    finetune_lr = model_config['finetune_lr']
    pretraining_epochs= model_config['pretraining_epochs']
    pretrain_lr = model_config['pretrain_lr']
    training_epochs = model_config['training_epochs']


    corruption_levels =sda_config['corruption_levels']

    #########################
    # PRETRAINING THE MODEL #
    #########################
    logger.info('Getting the pretraining functions....')
    pretraining_fns = sda.pretraining_functions(train_x=train_x,
                                                batch_size=batch_size)


    logger.info('Pre-training the model ...')
    start_time = time.clock()
    ## Pre-train layer-wise
    
    for i in xrange(sda.n_layers):
         # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []  # keep record of cost
            while not train_sets.is_finish():
                train_sets.make_partition_shared(train_xy)
                
                for batch_index in xrange(train_sets.cur_frame_num / batch_size):  # loop over mini-batches
                    #logger.info("Training For epoch %d and batch %d",epoch,batch_index)
                    curcost = pretraining_fns[i](index=batch_index,
                        corruption=corruption_levels[i],lr=pretrain_lr)
                    c.append(curcost)
                train_sets.read_next_partition_data()
            train_sets.initialize_read()
            logger.info("Pre-training layer %i, epoch %d, cost %f", i, epoch,numpy.mean(c))

    
    end_time = time.clock()
    logger.info('The PreTraing ran for %.2fm' % ((end_time - start_time) / 60.))

    

    ########################
    # FINETUNING THE MODEL #
    ########################

    try:
        valid_sets, valid_xy, valid_x, valid_y = read_dataset(data_spec['validation'])        
    except KeyError, e:
        #raise e
        logger.info("No validation set:Skiping Fine tunning");
        logger.info("Finshed")
        return


    # get the training, validation and testing function for the model
    #, test_model
    logger.info('Getting the finetuning functions')
    train_fn, validate_fn = sda.build_finetune_functions(
                train_x=train_x,train_y=train_y,valid_x=valid_x,valid_y=valid_y,
                batch_size=batch_size,learning_rate=finetune_lr)

    def valid_score():
        valid_error = []
        while not valid_sets.is_finish():
            valid_sets.make_partition_shared(valid_xy)
            n_valid_batches= valid_sets.cur_frame_num / batch_size;
            validation_losses = [validate_fn(i) for i in xrange(n_valid_batches)]
            valid_error.append(validation_losses)
            train_sets.read_next_partition_data()
        valid_sets.initialize_read();
        return numpy.mean(valid_error)


    
    logger.info('Finetunning the model..');
    
    #TODO include param in config
    # early-stopping parameters
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    
    start_time = time.clock()

    done_looping = False
    epoch = 0

    logger.debug('training_epochs = %d',training_epochs);

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1

        while not train_sets.is_finish():
            train_sets.make_partition_shared(train_xy)

            n_train_batches = train_sets.cur_frame_num / batch_size 
            patience = 10 * n_train_batches
            validation_frequency = min(n_train_batches, patience / 2)

            for minibatch_index in xrange(n_train_batches):
                minibatch_avg_cost = train_fn(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    this_validation_loss = validate_model()
                    logger.info('epoch %i, minibatch %i/%i, validation error %f %%',
                          epoch, minibatch_index + 1, n_train_batches,this_validation_loss * 100.)

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
    
                        #improve patience if loss improvement is good enough
                        if (this_validation_loss < best_validation_loss * improvement_threshold):
                            patience = max(patience, iter * patience_increase)
    
                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter
    
    
                if patience <= iter:
                    done_looping = True
                    break
            train_sets.read_next_partition_data()
        train_sets.initialize_read()

    end_time = time.clock()
    logger.info('The Fine tunning ran for %.2fm' % ((end_time - start_time) / 60.))
    logger.info('Optimization complete with best validation score of %f %%',best_validation_loss * 100)




if __name__ == '__main__':
    import sys
    setLogger(level="DEBUG");
    logger.info('Stating....');
    runSdA(sys.argv[1])
