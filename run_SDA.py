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
    pretraining_fns = sda.pretraining_functions(train_set_x=train_x,
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
    import os
    logger.info('The code for file ' + os.path.split(__file__)[1] +
                      ' ran for %.2fm' % ((end_time - start_time) / 60.))

    """

    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = sda.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)

    print '... finetunning the model'
    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    """


if __name__ == '__main__':
    import sys
    setLogger();
    logger.info('Stating....');
    runSdA(sys.argv[1])