import json,sys,theano
import numpy as np
from StringIO import StringIO

import logging
logger = logging.getLogger(__name__)

# convert an array to a string
def array_2_string(array):
	str_out = StringIO()
	np.savetxt(str_out, array)
	return str_out.getvalue()

# convert a string to an array
def string_2_array(string):
	str_in = StringIO(string)
	return np.loadtxt(str_in)

def _nnet2file(layers, set_layer_num = -1, filename='nnet.out', activation='sigmoid', start_layer = 0, withfinal=True, 
		input_factor = 0.0, factor=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]):
	n_layers = len(layers)
	nnet_dict = {}
	if set_layer_num == -1:
	   set_layer_num = n_layers - 1

	for i in range(start_layer, set_layer_num):
	   dict_a = str(i) + ' ' + activation + ' W'
	   if i == 0:
		   nnet_dict[dict_a] = array_2_string((1.0 - input_factor) * layers[i].params[0].get_value())
	   else:
		   nnet_dict[dict_a] = array_2_string((1.0 - factor[i-1]) * layers[i].params[0].get_value())
	   dict_a = str(i) + ' ' + activation + ' b'
	   nnet_dict[dict_a] = array_2_string(layers[i].params[1].get_value())
	
	if withfinal: 
		dict_a = 'logreg W'
		nnet_dict[dict_a] = array_2_string((1.0 - factor[-1]) * layers[-1].params[0].get_value())
		dict_a = 'logreg b'
		nnet_dict[dict_a] = array_2_string(layers[-1].params[1].get_value())
   
	with open(filename, 'wb') as fp:
		json.dump(nnet_dict, fp, indent=2, sort_keys = True)
		fp.flush() 
	logger.info('Dummped the neural_net model in %s',(filename))

def _file2nnet(layers, set_layer_num = -1, filename='nnet.in', activation='sigmoid', withfinal=True, factor=1.0):
	n_layers = len(layers)
	nnet_dict = {}
	if set_layer_num == -1:
		set_layer_num = n_layers - 1

	with open(filename, 'rb') as fp:
		nnet_dict = json.load(fp)
	for i in xrange(set_layer_num):
		dict_key = str(i) + ' ' + activation + ' W'
		layers[i].params[0].set_value(factor * np.asarray(string_2_array(nnet_dict[dict_key]), dtype=theano.config.floatX))
		dict_key = str(i) + ' ' + activation + ' b' 
		layers[i].params[1].set_value(np.asarray(string_2_array(nnet_dict[dict_key]), dtype=theano.config.floatX))

	if withfinal:
		dict_key = 'logreg W'
		layers[-1].params[0].set_value(np.asarray(string_2_array(nnet_dict[dict_key]), dtype=theano.config.floatX))
		dict_key = 'logreg b'
		layers[-1].params[1].set_value(np.asarray(string_2_array(nnet_dict[dict_key]), dtype=theano.config.floatX))
	logger.info('Loaded the neural_net model in %s',(filename))

def _cnn2file(conv_layers, filename='nnet.out', activation='sigmoid', withfinal=True, input_factor = 1.0, factor=1.0):
	n_layers = len(conv_layers)
	nnet_dict = {}
	for i in xrange(n_layers):
		conv_layer = conv_layers[i]
		filter_shape = conv_layer.filter_shape
		
		for next_X in xrange(filter_shape[0]):
			for this_X in xrange(filter_shape[1]):
				dict_a = 'W ' + str(i) + ' ' + str(next_X) + ' ' + str(this_X) 
				if i == 0:
					nnet_dict[dict_a] = array_2_string(input_factor * (conv_layer.W.get_value())[next_X, this_X])
				else:
					nnet_dict[dict_a] = array_2_string(factor * (conv_layer.W.get_value())[next_X, this_X])

		dict_a = 'b ' + str(i)
		nnet_dict[dict_a] = array_2_string(conv_layer.b.get_value())
	
	with open(filename, 'wb') as fp:
		json.dump(nnet_dict, fp, indent=2, sort_keys = True)
		fp.flush()
	logger.info('Dumped the conv_net model in %s',(filename))

def _file2cnn(conv_layers, filename='nnet.in', activation='sigmoid', withfinal=True, factor=1.0):
	n_layers = len(conv_layers)
	nnet_dict = {}

	with open(filename, 'rb') as fp:
		nnet_dict = json.load(fp)
	for i in xrange(n_layers):
		conv_layer = conv_layers[i]
		filter_shape = conv_layer.filter_shape
		W_array = conv_layer.W.get_value()

		for next_X in xrange(filter_shape[0]):
			for this_X in xrange(filter_shape[1]):
				dict_a = 'W ' + str(i) + ' ' + str(next_X) + ' ' + str(this_X)
				W_array[next_X, this_X, :, :] = factor * np.asarray(string_2_array(nnet_dict[dict_a]))

		conv_layer.W.set_value(W_array) 

		dict_a = 'b ' + str(i)
		conv_layer.b.set_value(np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
	logger.info('Loaded the conv_net model in %s',(filename))
