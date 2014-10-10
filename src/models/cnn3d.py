import theano,numpy
import theano.tensor as T

import json,os
from models import nnet,_array2string,_string2array

from layers.cnn3d import ConvLayer
from layers.logistic_sgd import LogisticRegression
from layers.mlp import HiddenLayer,DropoutHiddenLayer,_dropout_from_layer
from collections import OrderedDict
from io_modules.file_reader import read_dataset
from utils.plotter import plot

import logging
logger = logging.getLogger(__name__)
tensor5 = T.TensorType(theano.config.floatX, (False,)*5);

class CNN3DBase(nnet):
	def __init__(self, conv_layer_configs, hidden_layer_configs,l1_reg,l2_reg,max_col_norm):
		super(CNN3DBase, self).__init__()
		self.layers = []
		
		self.max_col_norm = max_col_norm
		self.l1_reg = l1_reg
		self.l2_reg = l2_reg

		self.x = tensor5('x')  
		self.y = T.ivector('y')

		self.conv_layer_num = len(conv_layer_configs) 	#counting number of convolution layers
		self.hidden_layer_num = len(hidden_layer_configs['hidden_layers'])
		self.mlp_layer_start = self.hidden_layer_num;
		self.mlp_layers = []
		self.conv_layers = []
	
	"""
	def save_cnn2dict(self):
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
		return cnn_dict;
	
	def save_mlp2dict(self,withfinal=True,max_layer_num=-1):
		raise NotImplementedError;
		
	def save(self,filename='nnet.out',max_layer_num=-1, withMLP = True, withfinal=True):
		#Dumping CNN Configuration
		cnn_dict = self.save_cnn2dict();
		if withMLP:
			#Dumping MLP Configuration
			mlp_dict = self.save_mlp2dict(withfinal,max_layer_num);
		else:
			mlp_dict = None

		nnet_dict = {};
		nnet_dict['cnn'] = cnn_dict;
		nnet_dict['mlp'] = mlp_dict;
	
		with open(filename, 'wb') as fp:
			json.dump(nnet_dict, fp, indent=2, sort_keys = True)
			fp.flush()
	
	def load_dict2cnn(self,cnn_dict):
		n_layers = self.conv_layer_num
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
				
	def load_dict2mlp(self,mlp_dict,max_layer_num=-1,withfinal=True):
		if max_layer_num == -1:
			max_layer_num = self.hidden_layer_num
			
		for i in xrange(max_layer_num):
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
		
	def load(self,filename='nnet.out',max_layer_num=-1, withMLP = True, withfinal=True):
		nnet_dict = {}
		with open(filename, 'rb') as fp:
			nnet_dict = json.load(fp)
			
		##Loading CNN Configuration
		self.load_dict2cnn(nnet_dict['cnn'])
			
		##Loading MLP Configuration
		try:
			mlp_dict  = nnet_dict['mlp'];
		except KeyError:
			return
		
		if withMLP and not mlp_dict is None:
			self.load_dict2mlp(mlp_dict,max_layer_num)
	
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
		for layer_idx in xrange(self.conv_layer_num):	
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
	"""

class CNN3D(CNN3DBase):
	""" Instantiation of Convolution neural network ... """
	def __init__(self, numpy_rng, theano_rng, batch_size, n_outs,conv_layer_configs, hidden_layer_configs, 
			conv_activation = T.nnet.sigmoid,hidden_activation = T.nnet.sigmoid,l1_reg=None,l2_reg=None,max_col_norm=None):

		super(CNN3D, self).__init__(conv_layer_configs, hidden_layer_configs,l1_reg,l2_reg,max_col_norm)
		if not theano_rng:
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
			            
		for i in xrange(self.conv_layer_num):		# construct the convolution layer
			if i == 0:  				#is_input layer
				input = self.x
			else:
				input = self.layers[-1].output #output of previous layer
			
			config = conv_layer_configs[i]
	
			conv_layer = ConvLayer(numpy_rng=numpy_rng, input=input,input_shape=config['input_shape'],
				filter_shape=config['filter_shape'],poolsize=config['poolsize'],
				activation = conv_activation)
			self.layers.append(conv_layer)
			self.conv_layers.append(conv_layer)
			if config['update']==True:	# only few layers of convolution layer are considered for updation
				self.params.extend(conv_layer.params)
				self.delta_params.extend(conv_layer.delta_params)

		hidden_layers = hidden_layer_configs['hidden_layers'];
		self.conv_output_dim = numpy.prod(config['output_shape'])
		adv_activation_configs = hidden_layer_configs['adv_activation'] 
		
		#flattening the last convolution output layer
		self.features = self.conv_layers[-1].output.flatten(2);
		self.features_dim = self.conv_output_dim;
		
		for i in xrange(self.hidden_layer_num):		# construct the hidden layer
			if i == 0:				# is first sigmoidla layer
				input_size = self.conv_output_dim
				layer_input = self.features
			else:
				input_size = hidden_layers[i - 1]	# number of hidden neurons in previous layers
				layer_input = self.layers[-1].output
			
			
			if adv_activation_configs is None:
				sigmoid_layer = HiddenLayer(rng=numpy_rng, input=layer_input,n_in=input_size, 
						n_out = hidden_layers[i], activation=hidden_activation);
						
			else:
				sigmoid_layer = HiddenLayer(rng=numpy_rng, input=layer_input,n_in=input_size, 
						n_out = hidden_layers[i]*adv_activation_configs['pool_size'], activation=hidden_activation,
						adv_activation_method = adv_activation_configs['method'],
						pool_size = adv_activation_configs['pool_size'],
						pnorm_order = adv_activation_configs['pnorm_order']);
						
						
			self.layers.append(sigmoid_layer)
			self.mlp_layers.append(sigmoid_layer)

			if config['update']==True:	# only few layers of hidden layer are considered for updation
                		self.params.extend(sigmoid_layer.params)
                		self.delta_params.extend(sigmoid_layer.delta_params)
           

		self.logLayer = LogisticRegression(input=self.layers[-1].output,n_in=hidden_layers[-1],n_out=n_outs)
		
		self.layers.append(self.logLayer)
		self.params.extend(self.logLayer.params)
		self.delta_params.extend(self.logLayer.delta_params)
		
		self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
		self.errors = self.logLayer.errors(self.y)
		self.output = self.logLayer.prediction()
		
		#regularization
		if self.l1_reg is not None:
			self.__l1Regularization__(self.hidden_layer_num*2);
		if self.l2_reg is not None:
			self.__l2Regularization__(self.hidden_layer_num*2);
		
	"""	
	def save_mlp2dict(self,withfinal=True,max_layer_num=-1):
		if max_layer_num == -1:
		   max_layer_num = self.hidden_layer_num
		mlp_dict = {}
		for i in range(max_layer_num):
			dict_a = str(i) +' W'
			mlp_dict[dict_a] = _array2string(self.mlp_layers[i].params[0].get_value())
			dict_a = str(i) + ' b'
			mlp_dict[dict_a] = _array2string(self.mlp_layers[i].params[1].get_value())

		if withfinal: 
			dict_a = 'logreg W'
			mlp_dict[dict_a] = _array2string(self.logLayer.params[0].get_value())
			dict_a = 'logreg b'
			mlp_dict[dict_a] = _array2string(self.logLayer.params[1].get_value())
		return mlp_dict
	"""

######################################## Dropout CNN ############################################
"""
class DropoutCNN(CNNBase):
	#Instantiation of Convolution neural network ... 
	def __init__(self, numpy_rng, theano_rng, batch_size, n_outs,conv_layer_configs, hidden_layer_configs, 
			use_fast=False,conv_activation = T.nnet.sigmoid,hidden_activation = T.nnet.sigmoid,
			l1_reg=None,l2_reg=None,max_col_norm=None,input_dropout_factor=0.0):

		super(DropoutCNN, self).__init__(conv_layer_configs,hidden_layer_configs,l1_reg,l2_reg,max_col_norm)
		self.input_dropout_factor = input_dropout_factor;
		
		self.dropout_layers = []
		if not theano_rng:
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
            
		for i in xrange(self.conv_layer_num):		# construct the convolution layer
			if i == 0:  							#is_input layer
				conv_input = self.x
				if self.input_dropout_factor > 0.0:
					dropout_conv_input = _dropout_from_layer(theano_rng, self.x,self.input_dropout_factor)
				else:
					dropout_conv_input = self.x;	
			else:
				conv_input = (1-conv_layer_configs[i-1]['dropout_factor'])*self.layers[-1].output #output of previous layer
				dropout_conv_input = self.dropout_layers[-1].dropout_output;
				
			config = conv_layer_configs[i]
			
			dropout_conv_layer = DropoutConvLayer(numpy_rng=numpy_rng, input=dropout_conv_input,
				input_shape=config['input_shape'],filter_shape=config['filter_shape'],poolsize=config['poolsize'],
				activation = conv_activation, use_fast = use_fast,dropout_factor=conv_layer_configs[i]['dropout_factor'])
			
			conv_layer = ConvLayer(numpy_rng=numpy_rng, input=conv_input,input_shape=config['input_shape'],
				filter_shape=config['filter_shape'],poolsize=config['poolsize'],activation = conv_activation,
				use_fast = use_fast, W = dropout_conv_layer.W, b = dropout_conv_layer.b)
			
				
			self.dropout_layers.append(dropout_conv_layer);
			self.layers.append(conv_layer)
			self.conv_layers.append(conv_layer)
			
			if config['update']==True:	# only few layers of convolution layer are considered for updation
				self.params.extend(dropout_conv_layer.params)
				self.delta_params.extend(dropout_conv_layer.delta_params)

		hidden_layers = hidden_layer_configs['hidden_layers'];
		self.conv_output_dim = config['output_shape'][1] * config['output_shape'][2] * config['output_shape'][3]
		adv_activation_configs = hidden_layer_configs['adv_activation'] 
		
		#flattening the last convolution output layer
		self.dropout_features = self.dropout_layers[-1].dropout_output.flatten(2);
		self.features = self.conv_layers[-1].output.flatten(2);
		self.features_dim = self.conv_output_dim;

		self.dropout_layers = [];
		self.dropout_factor = hidden_layer_configs['dropout_factor'];
		
		for i in xrange(self.hidden_layer_num):		# construct the hidden layer
			if i == 0:								# is first sigmoidal layer
				input_size = self.conv_output_dim
				dropout_layer_input = self.dropout_features
				layer_input = self.features
			else:
				input_size = hidden_layers[i - 1]	# number of hidden neurons in previous layers
				dropout_layer_input = self.dropout_layers[-1].dropout_output			
				layer_input = (1 - self.dropout_factor[i-1]) * self.layers[-1].output
				
			if adv_activation_configs is None:
				dropout_sigmoid_layer = DropoutHiddenLayer(rng=numpy_rng, input=layer_input,n_in=input_size, 
						n_out = hidden_layers[i], activation=hidden_activation,
						dropout_factor = self.dropout_factor[i]);
						
				sigmoid_layer = HiddenLayer(rng=numpy_rng, input=layer_input,n_in=input_size, 
						n_out = hidden_layers[i], activation=hidden_activation,
						W=dropout_sigmoid_layer.W, b=dropout_sigmoid_layer.b);
										
						
			else:
				dropout_sigmoid_layer = DropoutHiddenLayer(rng=numpy_rng, input=layer_input,n_in=input_size, 
						n_out = hidden_layers[i]*adv_activation_configs['pool_size'], activation=hidden_activation,
						adv_activation_method = adv_activation_configs['method'],
						pool_size = adv_activation_configs['pool_size'],
						pnorm_order = adv_activation_configs['pnorm_order'],
						dropout_factor = self.dropout_factor[i]);
						
				sigmoid_layer = HiddenLayer(rng=numpy_rng, input=layer_input,n_in=input_size, 
						n_out = hidden_layers[i]*adv_activation_configs['pool_size'], activation=hidden_activation,
						adv_activation_method = adv_activation_configs['method'],
						pool_size = adv_activation_configs['pool_size'],
						pnorm_order = adv_activation_configs['pnorm_order'],
						W=dropout_sigmoid_layer.W, b=dropout_sigmoid_layer.b);
						
			self.layers.append(sigmoid_layer)
			self.dropout_layers.append(dropout_sigmoid_layer)
			self.mlp_layers.append(sigmoid_layer)

			if config['update']==True:	# only few layers of hidden layer are considered for updation
						self.params.extend(dropout_sigmoid_layer.params)
						self.delta_params.extend(dropout_sigmoid_layer.delta_params)

		self.dropout_logLayer = LogisticRegression(input=self.dropout_layers[-1].dropout_output,n_in=hidden_layers[-1],n_out=n_outs)
		self.logLayer = LogisticRegression(
							input=(1 - self.dropout_factor[-1]) * self.layers[-1].output,
							n_in=hidden_layers[-1],n_out=n_outs,
							W=self.dropout_logLayer.W, b=self.dropout_logLayer.b)
		
		self.dropout_layers.append(self.dropout_logLayer)
		self.layers.append(self.logLayer)
		self.params.extend(self.dropout_logLayer.params)
		self.delta_params.extend(self.dropout_logLayer.delta_params)
		
		self.finetune_cost = self.dropout_logLayer.negative_log_likelihood(self.y)
		self.errors = self.logLayer.errors(self.y)
		self.output = self.logLayer.prediction()
		
		#regularization
		if self.l1_reg is not None:
			self.__l1Regularization__(self.hidden_layer_num*2);
		if self.l2_reg is not None:
			self.__l2Regularization__(self.hidden_layer_num*2);
			
			
			
	def save_mlp2dict(self,withfinal=True,max_layer_num=-1):
		if max_layer_num == -1:
		   max_layer_num = self.hidden_layer_num
		mlp_dict = {}
		for i in range(max_layer_num):
			dict_a = str(i) +' W'
			if i == 0:
				mlp_dict[dict_a] = _array2string((1.0 - self.input_dropout_factor) *self.mlp_layers[i].params[0].get_value())
			else:
				mlp_dict[dict_a] = _array2string((1.0 - self.dropout_factor[i - 1]) * self.mlp_layers[i].params[0].get_value())
			dict_a = str(i) + ' b'
			mlp_dict[dict_a] = _array2string(self.mlp_layers[i].params[1].get_value())	

		if withfinal: 
			dict_a = 'logreg W'
			mlp_dict[dict_a] = _array2string((1.0 - self.dropout_factor[-1])*self.logLayer.params[0].get_value())
			dict_a = 'logreg b'
			mlp_dict[dict_a] = _array2string(self.logLayer.params[1].get_value())
		return mlp_dict
"""
