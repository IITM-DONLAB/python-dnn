import cPickle,gzip,os,sys
import theano,numpy
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from layers.cnn import ConvLayer
from layers.logistic_sgd import LogisticRegression
from layers.mlp import HiddenLayer
from  theano.compat.python2x import OrderedDict
from io_modules.file_reader import read_dataset
from models import nnet
from utils.plotter import plot
import logging
logger = logging.getLogger(__name__)

class CNN(nnet):
	""" Instantiation of Convolution neural network ... """
	def __init__(self, numpy_rng, theano_rng, batch_size, n_outs,conv_layer_configs, hidden_layers_sizes, 
			use_fast=False,conv_activation = T.nnet.sigmoid,hidden_activation = T.nnet.sigmoid):

		super(CNN, self).__init__()
		self.layers = []

		#self.sparsity = sparsity
		#self.sparsity_weight = sparsity_weight
		#self.sparse_layer = sparse_layer

		if not theano_rng:	#if theano range not passed creating new random stream object
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

		self.x = T.tensor4('x')  
		self.y = T.ivector('y')
		self.conv_input_dim =  conv_layer_configs[0]['input_shape']
		self.conv_layer_num = len(conv_layer_configs) 	#counting number of convolution layers
        	self.hidden_layer_num = len(hidden_layers_sizes)
		self.conv_layers = []
		
		logger.info('Building convolution layers....')
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

		logger.info('Building Hidden layers....')
		for i in xrange(self.hidden_layer_num):		# construct the hidden layer
			if i == 0:				# is first sigmoidla layer
				input_size = self.conv_output_dim
			else:
				input_size = hidden_layers_sizes[i - 1]	# number of hidden neurons in previous layers
			layer_input = self.layers[-1].output
			sigmoid_layer = HiddenLayer(rng=numpy_rng, input=layer_input,n_in=input_size, 
						n_out = hidden_layers_sizes[i], activation=hidden_activation)
			self.layers.append(sigmoid_layer)

			if config['update']==True:	# only few layers of hidden layer are considered for updation
                		self.params.extend(sigmoid_layer.params)
                		self.delta_params.extend(sigmoid_layer.delta_params)
           
		logger.info('Building last logistic layer ....')
		self.logLayer = LogisticRegression(input=self.layers[-1].output,n_in=hidden_layers_sizes[-1],n_out=n_outs)
		
		self.layers.append(self.logLayer)
		self.params.extend(self.logLayer.params)
		self.delta_params.extend(self.logLayer.delta_params)
		
		self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
		self.errors = self.logLayer.errors(self.y)
		self.output = self.logLayer.prediction()
		
		self.features = self.conv_layers[-1].output;
		self.features_dim = self.conv_output_dim;
	
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
				
				
		
