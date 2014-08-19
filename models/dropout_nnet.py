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

from layers.logistic_sgd import LogisticRegression
from layers.mlp import HiddenLayer, DropoutHiddenLayer, _dropout_from_layer

class DNN_Dropout(object):

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10,
                 activation = T.nnet.sigmoid,
                 input_dropout_factor = 0,
                 dropout_factor = [0.2,0.2,0.2,0.2,0.2,0.2,0.2],
                 do_maxout = False, pool_size = 1,
                 max_col_norm = None, l1_reg = None, l2_reg = None):

        self.sigmoid_layers = []
        self.dropout_layers = []
        self.params = []
        self.delta_params   = []
        self.n_layers = len(hidden_layers_sizes)

        self.max_col_norm = max_col_norm
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        self.input_dropout_factor = input_dropout_factor
        self.dropout_factor = dropout_factor

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x') 
        self.y = T.ivector('y')

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer
            if i == 0:
                input_size = n_ins
                layer_input = self.x
                if input_dropout_factor > 0.0:
                    dropout_layer_input = _dropout_from_layer(theano_rng, self.x, input_dropout_factor)
                else:
                    dropout_layer_input = self.x
            else:
                input_size = hidden_layers_sizes[i - 1]
                layer_input = (1 - self.dropout_factor[i - 1]) * self.sigmoid_layers[-1].output
                dropout_layer_input = self.dropout_layers[-1].dropout_output

            if do_maxout == False:
                dropout_layer = DropoutHiddenLayer(rng=numpy_rng,
                                        input=dropout_layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation= activation,
                                        dropout_factor=self.dropout_factor[i])
                sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=activation,
                                        W=dropout_layer.W, b=dropout_layer.b)
            else:
                dropout_layer = DropoutHiddenLayer(rng=numpy_rng,
                                        input=dropout_layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i] * pool_size,
                                        activation= (lambda x: 1.0*x),
                                        dropout_factor=self.dropout_factor[i],
                                        do_maxout = True, pool_size = pool_size)
                sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i] * pool_size,
                                        activation= (lambda x: 1.0*x),
                                        W=dropout_layer.W, b=dropout_layer.b,
                                        do_maxout = True, pool_size = pool_size)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            self.dropout_layers.append(dropout_layer)
            self.params.extend(dropout_layer.params)
            self.delta_params.extend(dropout_layer.delta_params)
        # We now need to add a logistic layer on top of the MLP
        self.dropout_logLayer = LogisticRegression(
                                 input=self.dropout_layers[-1].dropout_output,
                                 n_in=hidden_layers_sizes[-1], n_out=n_outs)

        self.logLayer = LogisticRegression(
                         input=(1 - self.dropout_factor[-1]) * self.sigmoid_layers[-1].output,
                         n_in=hidden_layers_sizes[-1], n_out=n_outs,
                         W=self.dropout_logLayer.W, b=self.dropout_logLayer.b)

        self.dropout_layers.append(self.dropout_logLayer)
        self.sigmoid_layers.append(self.logLayer)
        self.params.extend(self.dropout_logLayer.params)
        self.delta_params.extend(self.dropout_logLayer.delta_params)

        # compute the cost
        self.finetune_cost = self.dropout_logLayer.negative_log_likelihood(self.y)
        self.errors = self.logLayer.errors(self.y)

        if self.l1_reg is not None:
            for i in xrange(self.n_layers):
                W = self.params[i * 2]
                self.finetune_cost += self.l1_reg * (abs(W).sum())

        if self.l2_reg is not None:
            for i in xrange(self.n_layers):
                W = self.params[i * 2]
                self.finetune_cost += self.l2_reg * T.sqr(W).sum()

    def build_finetune_functions(self, train_shared_xy, valid_shared_xy, batch_size):

        (train_set_x, train_set_y) = train_shared_xy
        (valid_set_x, valid_set_y) = valid_shared_xy

        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.fscalar('learning_rate')
        momentum = T.fscalar('momentum')

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = {}
        for dparam, gparam in zip(self.delta_params, gparams):
            updates[dparam] = momentum * dparam - gparam*learning_rate
        for dparam, param in zip(self.delta_params, self.params):
            updates[param] = param + updates[dparam]

        if self.max_col_norm is not None:
            for i in xrange(self.n_layers):
                W = self.params[i * 2]
                if W in updates:
                    updated_W = updates[W]
                    col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                    desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                    updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))

        train_fn = theano.function(inputs=[index, theano.Param(learning_rate, default = 0.0001),
              theano.Param(momentum, default = 0.5)],
              outputs=self.errors,
              updates=updates,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]})

        valid_fn = theano.function(inputs=[index],
              outputs=self.errors,
              givens={
                self.x: valid_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: valid_set_y[index * batch_size:
                                    (index + 1) * batch_size]})

        return train_fn, valid_fn

