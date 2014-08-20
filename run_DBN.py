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
from io_modules.file_io import read_dataset
from io_modules import setLogger
from utils.learn_rates import LearningRate

import logging
logger = logging.getLogger(__name__)

def preTraining(train_sets,train_xy,pretraining_fns,model_config,rbm_config):

    batch_size = model_config['batch_size'];
    pretrainingEpochs = model_config['pretraining_epochs']
    nPreTrainLayers = rbm_config['pretrained_layers']

    keep_layer_num=model_config['keep_layer_num']
    
    initialMomentum = model_config['initial_momentum']
    initMomentumEpochs = model_config['initial_momentum_epoch']
    finalMomentum = model_config['final_momentum']

    first_layer_gb = rbm_config['first_layer_gb']

    ## Pre-train layer-wise
    for i in range(keep_layer_num, nPreTrainLayers):
        if (first_layer_gb):
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
                for batch_index in xrange(train_sets.cur_frame_num / batch_size):  # loop over mini-batches
                    #logger.info("Training For epoch %d and batch %d",epoch,batch_index)
                    [reconstruction_cost, free_energy_cost] = pretraining_fns[i](index=batch_index,
                                                                             lr=pretrain_lr,
                                                                             momentum=momentum)
                    r_c.append(reconstruction_cost)
                    fe_c.append(free_energy_cost)
                train_sets.read_next_partition_data()
            train_sets.initialize_read()
            logger.info('Training layer %i, epoch %d, r_cost %f, fe_cost %f' % (i, epoch, numpy.mean(r_c), numpy.mean(fe_c)))


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

def getFeatures(dbn,data_spec_testing):
    out_function = dbn.getFeaturesFunction()
    test_sets, test_xy, test_x, test_y = read_dataset(data_spec_testing)
    while (not test_sets.is_finish()):
        data = out_function(test_sets.feat)
        test_sets.read_next_partition_data()
        #TODO write data



def runRBM(configFile):
    model_config = load_model(configFile)
    rbm_config = load_rbm_spec(model_config['rbm_nnet_spec'])
    data_spec =  load_data_spec(model_config['data_spec']);


    #generating Random
    numpy_rng = numpy.random.RandomState(rbm_config['random_seed'])
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))


    dbn = DBN(numpy_rng=numpy_rng, theano_rng = theano_rng, n_ins=rbm_config['n_ins'],
              hidden_layers_sizes=rbm_config['hidden_layers'],
              n_outs=rbm_config['n_outs'], first_layer_gb = rbm_config['first_layer_gb'])

    train_sets, train_xy, train_x, train_y = read_dataset(data_spec['training'])


    keep_layer_num=model_config['keep_layer_num']
    
    if keep_layer_num > 0:
    	#current_nnet = wdir + 'nnet.ptr.current'
        logger.info('Initializing model from ' + str(current_nnet) + '....')
        # load model
        #_file2nnet(dbn.sigmoid_layers, set_layer_num = keep_layer_num, filename = current_nnet, withfinal=False)

    logger.info('Getting the pretraining functions....')
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_x,
                                                 batch_size=model_config['batch_size'],
                                                 weight_cost = 0.0002)

    logger.info('Pre-training the model ...')
    start_time = time.clock()
    preTraining(train_sets,train_xy,pretraining_fns,model_config,rbm_config)
    end_time = time.clock()

    logger.info('The PreTraing ran for %.2fm' % ((end_time - start_time) / 60.))

    # save the pretrained nnet to file
    #_nnet2file(dbn.sigmoid_layers, filename=output_file, withfinal=True)
    

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
    train_fn, validate_fn = dbn.build_finetune_functions((train_x, train_y),
             (valid_x, valid_y), batch_size=model_config['batch_size'])
    
    try:
        finetune_method = model_config['finetune_method']
        finetune_config = model_config['finetune_rate'] 
        momentum = model_config['finetune_momentum']
        lrate = LearningRate.get_instance(finetune_method,finetune_config);        
    except KeyError, e:
        print("KeyMissing:"+str(e));
        print("Fine tunning Paramters Missing")
        sys.exit(2)

    
    logger.info('Finetunning the model..');
    
    start_time = time.clock()
    err=fineTunning(train_fn,validate_fn,train_sets,train_xy,
            valid_sets,valid_xy,lrate,momentum,model_config['batch_size']);
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
    test_fn = dbn.build_test_function((test_x, test_y), batch_size=model_config['batch_size'])

    logger.info('Starting Testing');
    test_pred,test_loss=testing(test_fn,test_sets,test_xy,model_config['batch_size'])
    logger.info('Optimization complete with best Test score of %f %%',test_loss * 100)


if __name__ == '__main__':
    import sys
    setLogger();
    runRBM(sys.argv[1])
