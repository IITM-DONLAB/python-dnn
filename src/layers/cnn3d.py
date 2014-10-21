import cPickle
import gzip
import os
import sys
import time
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal.downsample import DownsampleFactorMax
import theano.tensor.nnet.conv3d2d
from utils.utils import activation_to_txt
from utils.max_pool import max_pool_3d
T = theano.tensor
floatX = theano.config.floatX

class ConvLayer(object):
    """Pool Layer of a convolutional network """


    def __init__(self, numpy_rng, input, input_shape, filter_shape, poolsize, activation, W = None, b = None, border_mode = 'valid'):
        """
                input_shape = (batchsize, in_time, in_channels, in_height, in_width)
                filter_shape= (flt_channels, flt_time, in_channels, flt_height, flt_width)
        """

        assert input_shape[2] == filter_shape[2]
        self.input = input
        #print 'in > ',self.input.tag.test_value.shape, input_shape
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.poolsize = poolsize
        self.activation = activation
        if W is None:
            if activation_to_txt(activation) in ('relu', 'softplus'):
                norm_scale = 0.01
            else:
                fan_in = numpy.prod(filter_shape[1:])
                norm_scale = 2.0 * numpy.sqrt(1.0 / fan_in)
            w_values = numpy_rng.normal(loc=0, scale=norm_scale, size=filter_shape)
            w_values = numpy.asarray(w_values, dtype=theano.config.floatX)
            W = theano.shared(value=w_values, name='W')
        self.W = W
        if b is None:
            if activation_to_txt(activation) in ('relu', 'softplus'):
                b_values = numpy.ones((filter_shape[0],), dtype=theano.config.floatX)
            else:
                b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')
        self.b = b
        self.delta_W = theano.shared(value=numpy.zeros(filter_shape, dtype=theano.config.floatX), name='delta_W')
        self.delta_b = theano.shared(value=numpy.zeros_like(self.b.get_value(borrow=True), dtype=theano.config.floatX), name='delta_b')
        
        conv_out = T.nnet.conv3d2d.conv3d(signals=self.input, filters=self.W, signals_shape=input_shape, filters_shape=filter_shape, border_mode=border_mode)
        y_out = activation(conv_out + self.b.dimshuffle('x', 'x', 0, 'x', 'x'))
       
        pooled_out = max_pool_3d(input=y_out, ds=poolsize, ignore_border=True)
        self.output = pooled_out
        #print 'out > ',self.output.tag.test_value.shape
        self.params = [self.W, self.b]
        self.delta_params = [self.delta_W, self.delta_b]

