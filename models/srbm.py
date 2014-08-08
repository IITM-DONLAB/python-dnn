import os
import sys

import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from layers.logistic_sgd import LogisticRegression
from layers.mlp import HiddenLayer
from layers.rbm import RBM, GBRBM

class SRBM(object):

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10,
                 first_layer_gb = True):
        """ Stacked RBMs for DNN Pre-training """

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        #self.x = T.matrix('x')
        #self.x = T.ftensor4('x') 
        self.y = T.ivector('y')

        for i in xrange(self.n_layers):
            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
                layer_input = self.x
            else:
                input_size = hidden_layers_sizes[i - 1]
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            # the first layer could be Gaussian-Bernoulli RBM
            # other layers are Bernoulli-Bernoulli RBMs
            if i == 0 and first_layer_gb:
                rbm_layer = GBRBM(numpy_rng=numpy_rng,
                              theano_rng=theano_rng,
                              input=layer_input,
                              n_visible=input_size,
                              n_hidden=hidden_layers_sizes[i],
                              W=sigmoid_layer.W,
                              hbias=sigmoid_layer.b)
            else:
                rbm_layer = RBM(numpy_rng=numpy_rng,
                              theano_rng=theano_rng,
                              input=layer_input,
                              n_visible=input_size,
                              n_hidden=hidden_layers_sizes[i],
                              W=sigmoid_layer.W,
                              hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
                         input=self.sigmoid_layers[-1].output,
                         n_in=hidden_layers_sizes[-1], n_out=n_outs)

        self.sigmoid_layers.append(self.logLayer)
        self.params.extend(self.logLayer.params)

#    def pretraining_functions(self, train_set_x, batch_size, weight_cost):
    def pretraining_functions(self, train_set_x, batch_size, weight_cost):

        index = T.lscalar('index')  
        momentum = T.scalar('momentum')
        learning_rate = T.scalar('lr') 
        # number of mini-batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # start and end index of this mini-batch
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:
            r_cost, fe_cost, updates = rbm.get_cost_updates(batch_size, learning_rate,
                                                            momentum, weight_cost,
                                                            persistent=None)
            # compile the theano function
            fn = theano.function(inputs=[index,
                              theano.Param(learning_rate, default=0.0001),
                              theano.Param(momentum, default=0.5)],
                              outputs= [r_cost, fe_cost],
                              updates=updates,
                              givens={self.x: train_set_x[batch_begin:batch_end]})
            # append function to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

