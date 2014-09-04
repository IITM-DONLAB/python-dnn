import theano,numpy
import theano.tensor as T

import json,os
from models import nnet,_array2string,_string2array

from layers.cnn import ConvLayer
from layers.logistic_sgd import LogisticRegression
from layers.mlp import HiddenLayer,DropoutHiddenLayer,_dropout_from_layer
from collections import OrderedDict
from io_modules.file_reader import read_dataset
from utils.plotter import plot

import logging
logger = logging.getLogger(__name__)


class DropoutCNN(nnet):
	""" Instantiation of Convolution neural network ... """
	def __init__(self, numpy_rng, theano_rng, batch_size, n_outs,conv_layer_configs, hidden_layer_configs, 
			use_fast=False,conv_activation = T.nnet.sigmoid,hidden_activation = T.nnet.sigmoid):

		super(DropoutCNN, self).__init__()
		self.layers = []

		#self.sparsity = sparsity
		#self.sparsity_weight = sparsity_weight
		#self.sparse_layer = sparse_layer

		if not theano_rng:	#if theano range not passed creating new random stream object
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

		self.x = T.tensor4('x')  
		self.y = T.ivector('y')

		self.conv_layer_num = len(conv_layer_configs) 	#counting number of convolution layers
		hidden_layers = hidden_layer_configs['layers']
		self.hidden_layer_num = len(hidden_layers)
		self.mlp_layers = []
		self.conv_layers = []
		

		for i in xrange(self.conv_layer_num):		# construct the convolution layer
			if i == 0:  				#is_input layer
				input = self.x
				is_input_layer = True
			else:
				input = self.layers[-1].output #output of previous layer
				is_input_layer = False
			config = conv_layer_configs[i]
	
			conv_layer = ConvLayer(numpy_rng=numpy_rng, input=input,input_shape=config['input_shape'],
				filter_shape=config['filter_shape'],poolsize=config['poolsize'],
				flatten = config['flatten'],activation = conv_activation, use_fast = use_fast)
			self.layers.append(conv_layer)
			self.conv_layers.append(conv_layer)
			if config['update']==True:	# only few layers of convolution layer are considered for updation
				self.params.extend(conv_layer.params)
				self.delta_params.extend(conv_layer.delta_params)

		
		self.conv_output_dim = config['output_shape'][1] * config['output_shape'][2] * config['output_shape'][3]
		max_out_configs = hidden_layer_configs['max_out'] 

		self.dropout_layers = [];
		self.dropout_factor = hidden_layer_configs['dropout_factor'];
		for i in xrange(self.hidden_layer_num):		# construct the hidden layer
			if i == 0:				# is first sigmoidla layer
				input_size = self.conv_output_dim
				if self.dropout_factor > 0.0:
					dropout_layer_input = _dropout_from_layer(theano_rng, self.layers[-1].output, self.dropout_factor)
				else:
					dropout_layer_input = self.layers[-1].output
				layer_input = self.layers[-1].output
			else:
				input_size = hidden_layers[i - 1]	# number of hidden neurons in previous layers
				dropout_layer_input = self.dropout_layers[-1].dropout_output			
				layer_input = (1 - self.dropout_factor) * self.layers[-1].output
				
			if max_out_configs['method'] is None:
				dropout_sigmoid_layer = DropoutHiddenLayer(rng=numpy_rng, input=layer_input,n_in=input_size, 
						n_out = hidden_layers[i], activation=hidden_activation,
						maxout_method = max_out_configs['maxout_method'],
						pool_size = max_out_configs['pool_size'],
						pnorm_order = max_out_configs['pnorm_order'],
						dropout_factor = self.dropout_factor);
						
				sigmoid_layer = HiddenLayer(rng=numpy_rng, input=layer_input,n_in=input_size, 
						n_out = hidden_layers[i], activation=hidden_activation,
						maxout_method = max_out_configs['maxout_method'],
						pool_size = max_out_configs['pool_size'],
						pnorm_order = max_out_configs['pnorm_order'],
						W=dropout_sigmoid_layer.W, b=dropout_sigmoid_layer.b);
						
				
						
			else:
				dropout_sigmoid_layer = DropoutHiddenLayer(rng=numpy_rng, input=layer_input,n_in=input_size, 
						n_out = hidden_layers[i]*max_out_configs['pool_size'], activation=hidden_activation,
						maxout_method = max_out_configs['method'],
						pool_size = max_out_configs['pool_size'],
						pnorm_order = max_out_configs['pnorm_order'],
						dropout_factor = self.dropout_factor);
						
				sigmoid_layer = HiddenLayer(rng=numpy_rng, input=layer_input,n_in=input_size, 
						n_out = hidden_layers[i]*max_out_configs['pool_size'], activation=hidden_activation,
						maxout_method = max_out_configs['method'],
						pool_size = max_out_configs['pool_size'],
						pnorm_order = max_out_configs['pnorm_order'],
						W=dropout_sigmoid_layer.W, b=dropout_sigmoid_layer.b);
						
			self.layers.append(sigmoid_layer)
			self.dropout_layers.append(dropout_sigmoid_layer)
			self.mlp_layers.append(sigmoid_layer)

			if config['update']==True:	# only few layers of hidden layer are considered for updation
						self.params.extend(dropout_sigmoid_layer.params)
						self.delta_params.extend(dropout_sigmoid_layer.delta_params)

		self.dropout_logLayer = LogisticRegression(input=self.dropout_layers[-1].dropout_output,n_in=hidden_layers[-1],n_out=n_outs)
		self.logLayer = LogisticRegression(
							input=(1 - self.dropout_factor) * self.layers[-1].output,
							n_in=hidden_layers[-1],n_out=n_outs,
							W=self.dropout_logLayer.W, b=self.dropout_logLayer.b)
		
		self.dropout_layers.append(self.dropout_logLayer)
		self.layers.append(self.logLayer)
		self.params.extend(self.dropout_logLayer.params)
		self.delta_params.extend(self.dropout_logLayer.delta_params)
		
		self.finetune_cost = self.dropout_logLayer.negative_log_likelihood(self.y)
		self.errors = self.logLayer.errors(self.y)
		
		self.output = self.logLayer.prediction()
		self.features = self.conv_layers[-1].output;
		self.features_dim = self.conv_output_dim;

	def save(self,filename='nnet.out', withMLP = True, withfinal=True):
		#Dumping CNN Configuration
		n_layers = self.conv_layer_num
		cnn_dict = {}
		for i in xrange(n_layers):
			conv_layer = self.conv_layers[i]
			filter_shape = conv_layer.filter_shape
			for next_X in xrange(filter_shape[0]):
				for this_X in xrange(filter_shape[1]):
					dict_a = 'W ' + str(i) + ' ' + str(next_X) + ' ' + str(this_X) 
					cnn_dict[dict_a] = _array2string((conv_layer.W.get_value())[next_X, this_X])
	
			dict_a = 'b ' + str(i)
			cnn_dict[dict_a] = _array2string(conv_layer.b.get_value())
	
		if withMLP:
			#Dumping MLP Configuration
			n_layers = self.hidden_layer_num
			mlp_dict = {}
			for i in range(n_layers):
				dict_a = str(i) +' W'
				mlp_dict[dict_a] = _array2string(self.mlp_layers[i].params[0].get_value())
				dict_a = str(i) + ' b'
				mlp_dict[dict_a] = _array2string(self.mlp_layers[i].params[1].get_value())
	
			if withfinal: 
				dict_a = 'logreg W'
				mlp_dict[dict_a] = _array2string(self.logLayer.params[0].get_value())
				dict_a = 'logreg b'
				mlp_dict[dict_a] = _array2string(self.logLayer.params[1].get_value())
		else:
			mlp_dict = None
		nnet_dict = {};
		nnet_dict['cnn'] = cnn_dict;
		nnet_dict['mlp'] = mlp_dict;
	
		with open(filename, 'wb') as fp:
			json.dump(nnet_dict, fp, indent=2, sort_keys = True)
			fp.flush()
	
	
	def load(self,filename='nnet.out', withMLP = True, withfinal=True):
		nnet_dict = {}
		
		with open(filename, 'rb') as fp:
			nnet_dict = json.load(fp)
			
		##Loading CNN Configuration
		n_layers = self.conv_layer_num
		cnn_dict  = nnet_dict['cnn'];
		for i in xrange(n_layers):
			conv_layer = self.conv_layers[i]
			filter_shape = conv_layer.filter_shape
			W_array = conv_layer.W.get_value()
			
			for next_X in xrange(filter_shape[0]):
				for this_X in xrange(filter_shape[1]):
					dict_a = 'W ' + str(i) + ' ' + str(next_X) + ' ' + str(this_X)
					W_array[next_X, this_X, :, :] = numpy.asarray(_string2array(cnn_dict[dict_a]))
					
			conv_layer.W.set_value(W_array) 
			dict_a = 'b ' + str(i)
			conv_layer.b.set_value(numpy.asarray(_string2array(cnn_dict[dict_a]),
				dtype=theano.config.floatX))
			
		##Loading MLP Configuration
		try:
			mlp_dict  = nnet_dict['mlp'];
		except KeyError:
			return
		
		if withMLP and not mlp_dict is None:
			n_layers = self.hidden_layer_num
			for i in xrange(n_layers):
				dict_key = str(i) + ' W'
				self.mlp_layers[i].params[0].set_value(numpy.asarray(_string2array(mlp_dict[dict_key]),
					dtype=theano.config.floatX))
				dict_key = str(i) + ' b' 
				self.mlp_layers[i].params[1].set_value(numpy.asarray(_string2array(mlp_dict[dict_key]),
					dtype=theano.config.floatX))
			if withfinal:
				dict_key = 'logreg W'
				self.logLayer.params[0].set_value(numpy.asarray(_string2array(mlp_dict[dict_key]),
					dtype=theano.config.floatX))
				dict_key = 'logreg b'
				self.logLayer.params[1].set_value(numpy.asarray(_string2array(mlp_dict[dict_key]),
					dtype=theano.config.floatX))
	
	def getLayerOutFunction(self,idx):
		in_x = self.x.type('in_x');
		fn = theano.function(inputs=[in_x],outputs=self.layers[idx].output,
			givens={self.x: in_x})
		return fn
	
	def plot_layer_output(self,plot_spec,plot_path,max_images=10):
		#default all nodes set to value 1
		#inp = numpy.random.random(self.conv_input_dim).astype(theano.config.floatX);
		batch_size = plot_spec['batch_size'];
		plot_path = plot_path +os.sep +'layer_%d'+os.sep +'batch_%d'+os.sep+'img_%d.png'
		for layer_idx in xrange(self.conv_layer_num-1):	
			img_plot_remaining = max_images;
			layer_out_fn = self.getLayerOutFunction(layer_idx);
			logger.info('Plotting the layer %d'%layer_idx);
			file_reader =read_dataset(plot_spec,pad_zeros=True)[0];
			while not file_reader.is_finish():
				for batch_index in xrange(file_reader.cur_frame_num/batch_size):
					s_idx = batch_index * batch_size; e_idx = s_idx + batch_size
					data = layer_out_fn(file_reader.feat[s_idx:e_idx])
					e_idx= min(file_reader.cur_frame_num - file_reader.num_pad_frames,s_idx+batch_size);
					img_plot_remaining = plot(data[s_idx:e_idx],plot_path,layer_idx,batch_index,img_plot_remaining);
					if img_plot_remaining == 0:
						break;
				if img_plot_remaining == 0:
					break;
				file_reader.read_next_partition_data(pad_zeros=True);
