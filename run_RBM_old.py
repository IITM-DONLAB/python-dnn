# Copyright 2013    Yajie Miao    Carnegie Mellon University

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

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from models.srbm import SRBM

from io_func.pfile_io import read_data_args, read_dataset
from io_func.model_io import _nnet2file, _file2nnet, log

from utils.utils import parse_arguments, parse_activation

if __name__ == '__main__':

    import sys

    arg_elements=[]
    for i in range(1, len(sys.argv)):
        arg_elements.append(sys.argv[i])
    arguments = parse_arguments(arg_elements)

    if (not arguments.has_key('train_data')) or (not arguments.has_key('nnet_spec')) or (not arguments.has_key('wdir')) or (not arguments.has_key('output_file')):
        print "Error: the mandatory arguments are: --train-data --nnet-spec --wdir --output-file"
        exit(1)

    train_data_spec = arguments['train_data']
    nnet_spec = arguments['nnet_spec']
    wdir = arguments['wdir']
    output_file = arguments['output_file']

    gbrbm_learning_rate = 0.005
    learning_rate = 0.08
    batch_size=128
    epochs=10
    if arguments.has_key('gbrbm_learning_rate'):
        gbrbm_learning_rate = float(arguments['gbrbm_learning_rate'])
    if arguments.has_key('learning_rate'):
        learning_rate = float(arguments['learning_rate'])
    if arguments.has_key('batch_size'):
        batch_size = int(arguments['batch_size'])
    if arguments.has_key('epoch_number'):
        epochs = int(arguments['epoch_number'])

    # momentum; more complicated than dnn 
    initial_momentum = 0.5     # initial momentum 
    final_momentum = 0.9       # final momentum
    initial_momentum_epoch = 5 # for how many epochs do we use initial_momentum
    if arguments.has_key('momentum'):
        momentum_elements = arguments['momentum'].split(':')
        if len(momentum_elements) != 3:
            print "Error: momentum string should have 3 values, e.g., 0.5:0.9:5"
            exit(1)
        initial_momentum = float(momentum_elements[0])
        final_momentum = float(momentum_elements[1])
        initial_momentum_epoch = int(momentum_elements[2])

    dataset, dataset_args = read_data_args(train_data_spec)

    nnet_layers = nnet_spec.split(":")
    n_ins = int(nnet_layers[0])
    hidden_layers_sizes = []
    for i in range(1, len(nnet_layers)-1):
        hidden_layers_sizes.append(int(nnet_layers[i]))
    n_outs = int(nnet_layers[-1])

    ptr_layer_number = len(hidden_layers_sizes)
    if arguments.has_key('ptr_layer_number'):
        ptr_layer_number = int(arguments['ptr_layer_number'])

    first_layer_gb = True
    if arguments.has_key('first_layer_type'):
        if arguments['first_layer_type'] == 'bb':
            first_layer_gb = False

    # if initialized with current
    keep_layer_num=0
    current_nnet = wdir + 'nnet.ptr.current'
  
    train_sets, train_xy, train_x, train_y = read_dataset(dataset, dataset_args)

    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
    log('> ... building the model')
    # construct the stacked denoising autoencoder class
    srbm = SRBM(numpy_rng=numpy_rng, theano_rng = theano_rng, n_ins=n_ins,
              hidden_layers_sizes=hidden_layers_sizes,
              n_outs=n_outs, first_layer_gb = first_layer_gb)

    if keep_layer_num > 0:
        log('> ... initializing model from ' + current_nnet)
        _file2nnet(srbm.sigmoid_layers, set_layer_num = keep_layer_num, filename = current_nnet, withfinal=False)

    # PRETRAINING THE MODEL #
    log('> ... getting the pretraining functions')
    pretraining_fns = srbm.pretraining_functions(train_set_x=train_x,
                                                 batch_size=batch_size,
                                                 k = 1, weight_cost = 0.0002)

    log('> ... pre-training the model')
    start_time = time.clock()

    ## Pre-train layer-wise
    for i in range(keep_layer_num, ptr_layer_number):
        if (srbm.rbm_layers[i].is_gbrbm()):
            pretrain_lr = gbrbm_learning_rate
        else:
            pretrain_lr = learning_rate
        # go through pretraining epochs
        momentum = initial_momentum
        for epoch in xrange(epochs):
            # go through the training set
            if (epoch == initial_momentum_epoch):
                momentum = final_momentum

            r_c, fe_c = [], []  # keep record of reconstruction and free-energy cost
            while (not train_sets.is_finish()):
                train_sets.load_next_partition(train_xy)
                for batch_index in xrange(train_sets.cur_frame_num / batch_size):  # loop over mini-batches
                    [reconstruction_cost, free_energy_cost] = pretraining_fns[i](index=batch_index,
                                                                             lr=pretrain_lr,
                                                                             momentum=momentum)
                    r_c.append(reconstruction_cost)
                    fe_c.append(free_energy_cost)
            train_sets.initialize_read()
            log('> Pre-training layer %i, epoch %d, r_cost %f, fe_cost %f' % (i, epoch, numpy.mean(r_c), numpy.mean(fe_c)))

    # save the pretrained nnet to file
    _nnet2file(srbm.sigmoid_layers, filename=output_file, withfinal=True)

    end_time = time.clock()

    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
                      ' ran for %.2fm' % ((end_time - start_time) / 60.))

