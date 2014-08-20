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
from utils.load_conf import load_model,load_sda_spec,load_data_spec
from io_modules.file_io import read_dataset
from io_modules import setLogger
from utils.learn_rates import LearningRate
from models.sda import SDA


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


def fineTunning(train_fn,validate_fn,train_sets,train_xy,valid_sets,valid_xy,lrate,momentum,batch_size):

    def valid_score():
        valid_error = [] 
        while not valid_sets.is_finish():
            valid_sets.make_partition_shared(valid_xy)
            n_valid_batches= valid_sets.cur_frame_num / batch_size;
            validation_losses = [validate_fn(i) for i in xrange(n_valid_batches)]
            valid_error.append(validation_losses)
            valid_sets.read_next_partition_data()
            logger.debug("Valid Error (upto curr part) = %f",numpy.mean(valid_error))
        valid_sets.initialize_read();
        return numpy.mean(valid_error);

    best_validation_loss=float('Inf')
    while (lrate.get_rate() != 0):
        train_error = []
        while not train_sets.is_finish():
            train_sets.make_partition_shared(train_xy)
            for batch_index in xrange(train_sets.cur_frame_num / batch_size):  # loop over mini-batches
                train_error.append(train_fn(index=batch_index,
                    learning_rate = lrate.get_rate(), momentum = momentum))
                logger.debug('Training batch %d error %f',batch_index, numpy.mean(train_error))
            train_sets.read_next_partition_data()
        logger.info('Fine Tunning:epoch %d, training error %f',lrate.epoch, numpy.mean(train_error));
        train_sets.initialize_read()

        valid_error = valid_score()
        if valid_error < best_validation_loss:
            best_validation_loss=valid_error
        logger.info('Fine Tunning:epoch %d, validation error %f',lrate.epoch, valid_error);
        lrate.get_next_rate(current_error = 100 * valid_error)

    logger.info('Best validation error %f',best_validation_loss)
    return best_validation_loss

def testing(test_fn,test_sets,test_xy,batch_size):
    test_error  = []
    test_output = numpy.array([],int);
    while not test_sets.is_finish():
        test_sets.make_partition_shared(test_xy)
        n_test_batches= test_sets.cur_frame_num / batch_size;
        for i in xrange(n_test_batches):
            pred, err = test_fn(i)
            test_error.append(err)
            test_output=numpy.append(test_output,pred)
        test_sets.read_next_partition_data()
        logger.debug("Test Error (upto curr part) = %f",numpy.mean(test_error))
    test_sets.initialize_read();
    return test_output,numpy.mean(test_error);

def getFeatures(sda,data_spec_testing):
    out_function = sda.getFeaturesFunction()
    test_sets, test_xy, test_x, test_y = read_dataset(data_spec_testing)
    while (not test_sets.is_finish()):
        data = out_function(test_sets.feat)
        test_sets.read_next_partition_data()
        #TODO write data




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



    corruption_levels =sda_config['corruption_levels']

    #########################
    # PRETRAINING THE MODEL #
    #########################
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

    try:
        valid_sets, valid_xy, valid_x, valid_y = read_dataset(data_spec['validation'])        
    except KeyError:
        #raise e
        logger.info("No validation set:Skiping Fine tunning");
        logger.info("Finshed")
        sys.exit(0)


    # get the training, validation function for the model
    logger.info('Getting the finetuning functions')
    train_fn, validate_fn = sda.build_finetune_functions((train_x, train_y),
             (valid_x, valid_y), batch_size=batch_size)
    
    try:
        finetune_method = model_config['finetune_method']
        finetune_config = model_config['finetune_rate'] 
        momentum = model_config['finetune_momentum']
        lrate = LearningRate.get_instance(finetune_method,finetune_config);        
    except KeyError, e:
        print(str(e));
        print("Fine tunning Paramters Missing")
        sys.exit(2)

    
    logger.info('Finetunning the model..');
    
    start_time = time.clock()
    err=fineTunning(train_fn,validate_fn,train_sets,train_xy,
            valid_sets,valid_xy,lrate,momentum,batch_size);
    end_time = time.clock()

    logger.info('The Fine tunning ran for %.2fm' % ((end_time - start_time) / 60.))
    logger.info('Optimization complete with best validation score of %f %%',err * 100)



    try:
        test_sets, test_xy, test_x, test_y = read_dataset(data_spec['testing'])        
    except KeyError:
        #raise e
        logger.info("No testing set:Skiping Testing");
        logger.info("Finshed")
        sys.exit(0)


    # get the testing function for the model
    logger.info('Getting the Test function')
    test_fn = sda.build_test_function((test_x, test_y), batch_size=batch_size)

    logger.info('Starting Testing');
    test_pred,test_loss=testing(test_fn,test_sets,test_xy,batch_size)
    logger.info('Optimization complete with best Test score of %f %%',test_loss * 100)

    #print test_pred


if __name__ == '__main__':
    setLogger(level="INFO");
    logger.info('Stating....');
    runSdA(sys.argv[1]);
    sys.exit(0)
