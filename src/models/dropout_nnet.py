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

import numpy,json
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from layers.logistic_sgd import LogisticRegression
from layers.mlp import HiddenLayer, DropoutHiddenLayer, _dropout_from_layer

from models import nnet,_array2string,_string2array


class DNN_Dropout(nnet):

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10,
                 activation = T.nnet.sigmoid, input_dropout_factor = 0,
                 dropout_factor = [0.2,0.2,0.2,0.2,0.2,0.2,0.2],
                 adv_activation = None, max_col_norm = None,
                 l1_reg = None, l2_reg = None):

        super(DNN_Dropout, self).__init__()

        self.layers = []
        self.dropout_layers = []
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
                layer_input = (1 - self.dropout_factor[i - 1]) * self.layers[-1].output
                dropout_layer_input = self.dropout_layers[-1].dropout_output
			
            if not adv_activation  is None:
                dropout_layer = DropoutHiddenLayer(rng=numpy_rng,
                                        input=dropout_layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i] * adv_activation['pool_size'],
                                        activation= activation,
                                        adv_activation_method = adv_activation['method'],
                                        pool_size = adv_activation['pool_size'],
                                        pnorm_order = adv_activation['pnorm_order'],
                                        dropout_factor=self.dropout_factor[i])
                sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i] * adv_activation['pool_size'],
                                        activation=activation,
                                        adv_activation_method = adv_activation['method'],
                                        pool_size = adv_activation['pool_size'],
                                        pnorm_order = adv_activation['pnorm_order'],
                                        W=dropout_layer.W, b=dropout_layer.b)
            else:
                dropout_layer = DropoutHiddenLayer(rng=numpy_rng,
                                        input=dropout_layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation= activation,
                                        dropout_factor=self.dropout_factor[i])
                sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i] ,
                                        activation= activation,
                                        W=dropout_layer.W, b=dropout_layer.b)
                                        
            # add the layer to our list of layers
            self.layers.append(sigmoid_layer)
            self.dropout_layers.append(dropout_layer)
            self.params.extend(dropout_layer.params)
            self.delta_params.extend(dropout_layer.delta_params)
            
        # We now need to add a logistic layer on top of the MLP
        self.dropout_logLayer = LogisticRegression(
                                 input=self.dropout_layers[-1].dropout_output,
                                 n_in=hidden_layers_sizes[-1], n_out=n_outs)

        self.logLayer = LogisticRegression(
                         input=(1 - self.dropout_factor[-1]) * self.layers[-1].output,
                         n_in=hidden_layers_sizes[-1], n_out=n_outs,
                         W=self.dropout_logLayer.W, b=self.dropout_logLayer.b)

        self.dropout_layers.append(self.dropout_logLayer)
        self.layers.append(self.logLayer)
        self.params.extend(self.dropout_logLayer.params)
        self.delta_params.extend(self.dropout_logLayer.delta_params)

        # compute the cost
        self.finetune_cost = self.dropout_logLayer.negative_log_likelihood(self.y)
        self.errors = self.logLayer.errors(self.y)

        self.output = self.logLayer.prediction();
        self.features = self.layers[-2].output;
        self.features_dim = self.layers[-2].n_out

        if self.l1_reg is not None:
            self.__l1Regularization__();

        if self.l2_reg is not None:
            self.__l2Regularization__();


    def save(self,filename,start_layer = 0,max_layer_num = -1,withfinal=True):
        nnet_dict = {}
        if max_layer_num == -1:
           max_layer_num = self.n_layers

        for i in range(start_layer, max_layer_num):
           dict_a = str(i) + ' W'
           if i == 0:
               nnet_dict[dict_a] = _array2string((1.0 - self.input_dropout_factor) * (
                self.layers[i].params[0].get_value()))
           else:
               nnet_dict[dict_a] = _array2string((1.0 - self.dropout_factor[i - 1])* (
                self.layers[i].params[0].get_value()))
           dict_a = str(i) + ' b'
           nnet_dict[dict_a] = _array2string(self.layers[i].params[1].get_value())

        if withfinal: 
            dict_a = 'logreg W'
            nnet_dict[dict_a] = _array2string((1.0 - self.dropout_factor[-1])* (
                self.logLayer.params[0].get_value()))
            dict_a = 'logreg b'
            nnet_dict[dict_a] = _array2string(self.logLayer.params[1].get_value())
   
        with open(filename, 'wb') as fp:
            json.dump(nnet_dict, fp, indent=2, sort_keys = True)
            fp.flush()

    def load(self,filename,start_layer = 0,max_layer_num = -1,withfinal=True):
        nnet_dict = {}
        if max_layer_num == -1:
            max_layer_num = self.n_layers

        with open(filename, 'rb') as fp:
            nnet_dict = json.load(fp)
        
        for i in xrange(max_layer_num):
            dict_key = str(i) + ' W'
            self.layers[i].params[0].set_value(numpy.asarray(_string2array(nnet_dict[dict_key]),
                dtype=theano.config.floatX))
            dict_key = str(i) + ' b' 
            self.layers[i].params[1].set_value(numpy.asarray(_string2array(nnet_dict[dict_key]),
                dtype=theano.config.floatX))

        if withfinal:
            dict_key = 'logreg W'
            self.logLayer.params[0].set_value(numpy.asarray(_string2array(nnet_dict[dict_key]),
                dtype=theano.config.floatX))
            dict_key = 'logreg b'
            self.logLayer.params[1].set_value(numpy.asarray(_string2array(nnet_dict[dict_key]),
                dtype=theano.config.floatX))


