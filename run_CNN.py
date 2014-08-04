import cPickle, gzip, os, sys, time

#import numpy

#import theano
#import theano.tensor as T
#from theano.tensor.shared_randomstreams import RandomStreams

from utils.load_conf import load_model,load_conv_spec,load_mlp_spec,load_data_spec

#from io.file_io import read_data_args, read_dataset
from utils.learn_rates import LearningRateExpDecay,LearningRateConstant
#from utils.utils import parse_conv_spec, parse_lrate, parse_arguments


if __name__ == '__main__':
	model_config = load_model(sys.argv[1])
	
	# learning rate
	if model_config['l_rate_method'] =='E':
		lrate = LearningRateExpDecay(model_config['l_rate'])
	else:
		lrate =  LearningRateConstant(model_config['l_rate'])

	conv_config,convlayer_config = load_conv_spec(model_config['conv_nnet_spec'],model_config['batch_size'],
				model_config['input_shape'])
	
	mlp_config = load_mlp_spec(model_config['hidden_nnet_spec']);
	
	data_spec =  load_data_spec(model_config['data_spec']);


	
