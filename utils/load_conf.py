from json import load as jsonLoad
from utils import makeAbsolute,pathSep

import logging
logger = logging.getLogger(__name__)

def load_json(input_file):
	with open(input_file) as data_file:
		data = jsonLoad(data_file)
		try:
			data.pop('comment')
		except KeyError, e:
			pass
	return data;

def load_model(input_file,nnetType=None):
	logger.info("Loading model properties from %s ",input_file)
	data = load_json(input_file)

	#checking nnetType
	if nnetType==None:
		try:
			nnetType=data['nnetType']
		except KeyError, e:
			logger.critical(" 'nnetType' is missing in model properties file..")
			exit(1)
	else :
		if data.has_key('nnetType') and nnetType!=data['nnetType']:
			logger.critical(" 'nnetType' is not Matching..")
			exit(1)

	if checkConfig(data,nnetType):
		logger.critical(" the mandatory arguments are missing in model properties file..")
		exit(1)

	specPaths=['data_spec','conv_nnet_spec','hidden_nnet_spec',
				'rbm_nnet_spec','sda_nnet_spec']

	data = correctPath(data,specPaths,input_file);

	outputFiles=['output_file','conv_output_file',
				'hidden_output_file','export_path']

	data = correctPath(data,outputFiles,data['wdir']+pathSep);

	#init Default Values or update from Json.
	if nnetType == 'CNN':
		data = initModelCNN(data)
	elif nnetType == 'RBM':
		data = initModelRBM(data)
	elif nnetType == 'SDA':
		data = initModelSDA(data)
	else:	
		logger.error('Unknown nnetType')
		exit(1)
	
	#__debugPrintData__(data,'model');
	return data;

def correctPath(data,keys,basePath):
	for key in keys:
		if data.has_key(key):
			data[key] = makeAbsolute(data[key],basePath)
	return data

def checkConfig(data,nnetType):
	if not data.has_key('data_spec'): 
		logger.error('Missing Key in JSON :data_spec')
		return False
	if not data.has_key('wdir'):
		logger.error('Missing Key in JSON :wdir')
		return False
	if nnetType == 'CNN':
		requiredKeys=['conv_output_file','hidden_output_file','conv_nnet_spec', \
		'hidden_nnet_spec','input_shape','n_outs']
	elif nnetType == 'RBM':
		requiredKeys=['rbm_nnet_spec','output_file']
	elif nnetType == 'SDA':
		requiredKeys = ['sda_nnet_spec','output_file']
	else :
		logger.error('Unknown nnet Type')
		return False

	if isKeysPresents(data,requiredKeys):
		return False

	return True

def isKeysPresents(data,requiredKeys):
	for key in requiredKeys:
		if not data.has_key(key):
			logger.error('Missing Key in JSON :'+str(key))
			return False 
	return True
	

def load_data_spec(input_file):
	logger.info("Loading data specification properties from %s..",input_file)
	data = load_json(input_file);
	for x in ['training','testing','validation']:
		if not data.has_key(x):
			continue;
		if not data[x].has_key('keep_flatten') or not type(data[x]['keep_flatten']) is bool:
			data[x]['keep_flatten']=False
	return data


def load_mlp_spec(input_file):
	logger.info("Loading mlp properties from %s ...",input_file)
	return load_json(input_file);

	
#############################################################################
#CNN
#############################################################################
def initModelCNN(data):
	if not data.has_key('batch_size') or not type(data['batch_size']) is int:
		data['batch_size']=256
	if not data.has_key('momentum') or not type(data['momentum']) is float:
		data['momentum']=0.5
	
	if not data.has_key('l_rate_method'):
		data['l_rate_method']="C"
	
	if not data.has_key('l_rate'):
		lrate_config=dict()
		if data['l_rate_method'] == "C":
			lrate_config['learning_rate'] = 0.08
			lrate_config['epoch_num'] = 15
		else:
			lrate_config['start_rate'] = 0.08
			lrate_config['scale_by'] = 0.08
			lrate_config['min_derror_decay_start'] = 0.05
			lrate_config['min_derror_stop'] = 0.05
			lrate_config['min_epoch_decay_start'] = 15
			lrate_config['init_error'] = 100
		data['l_rate']=lrate_config

	return data

def load_conv_spec(input_file,batch_size,input_shape):
	logger.info("Loading convnet properties from %s ...",input_file)	
	data = load_json(input_file)  
	if not data.has_key('cnn'):
		logger.critical("CNN configuration is not present in " + str(input_file))
		exit(1)	
	cnn_data = data['cnn'];
	layer_configs=cnn_data.pop('layers');
	conv_configs = cnn_data;
	if len(layer_configs)==0:
		print "Error: No convnet configuration avaialable.."
		exit(1)	
	prev_map_number = 1;
	for layer_index in range(0,len(layer_configs)):
		if layer_index==0:
			layer_configs[layer_index]['input_shape']=[batch_size];
			layer_configs[layer_index]['input_shape'].extend(input_shape);
			prev_map_number = input_shape[0];			
		else:
			layer_configs[layer_index]['input_shape']=[batch_size];
			layer_configs[layer_index]['input_shape'].extend(input_shape);

		current_map_number = layer_configs[layer_index]['num_filters']
		layer_configs[layer_index]['filter_shape']=[current_map_number,prev_map_number];
		layer_configs[layer_index]['filter_shape'].extend(layer_configs[layer_index]['	']);
		
		layer_configs[layer_index]['output_shape'] = [batch_size,current_map_number];
		if not len(layer_configs[layer_index]['input_shape'][2:]) == len(layer_configs[layer_index]['convmat_dim']):
			logger.error("Input shape and convolution matrix dimension are not matching on layer %d ",layer_index+1)
		input_shape=[current_map_number];
		for inp,wdim,pool in zip(layer_configs[layer_index]['input_shape'][2:],layer_configs[layer_index]['convmat_dim'],
				layer_configs[layer_index]['poolsize']):
			outdim = (inp-wdim+1)/pool
			layer_configs[layer_index]['output_shape'].append(outdim)
			input_shape.append(outdim);
	
		prev_map_number = current_map_number
	if not data.has_key('cnn'):
		logger.critical("mlp configuration is not present in " + str(input_file))
		exit(1)	
	mlp_configs = data['mlp'];	
	return (conv_configs,layer_configs,mlp_configs)	

#############################################################################
#DBN/RBM
#############################################################################



def load_rbm_spec(input_file):
	logger.info("Loading net properties from %s ..",input_file)	
	data = load_json(input_file)

	if not data.has_key('hidden_layers') or not type(data['hidden_layers']) is list:
		logger.critical(" hidden_layers is not present (or not a list) in " + str(input_file))
		exit(1)

	if not data.has_key('n_ins') or not type(data['n_ins']) is int:
		logger.critical(" n_ins is not present (or not a int) in " + str(input_file))
		exit(1)

	if not data.has_key('n_outs') or not type(data['n_outs']) is int:
		logger.critical(" n_outs is not present (or not a int) in " + str(input_file))
		exit(1)


	if not data.has_key('pretrained_layers') or not type(data['pretrained_layers']) is int:
		data['pretrained_layers'] = len(data['hidden_layers'])
	elif data['pretrained_layers'] > (len(data['hidden_layers'])):
		data['pretrained_layers'] = len(data['hidden_layers'])

	first_layer_gb = True
	if data.has_key('first_layer_type') and data['first_layer_type'] == 'bb':
		first_layer_gb = False
	data['first_layer_gb'] = first_layer_gb

	if not data.has_key('random_seed') or not type(data['random_seed']) is int:
		data['random_seed'] = None

	#__debugPrintData__(data,'rbm');
	
	return (data)



def initModelRBM(data):

	#default values:

	gbrbm_learning_rate = 0.005
	learning_rate = 0.08
	batch_size=128
	epochs=10
	keep_layer_num=0

	# momentum; more complicated than dnn 
	initial_momentum = 0.5	 # initial momentum 
	final_momentum = 0.9	   # final momentum
	initial_momentum_epoch = 5 # for how many epochs do we use initial_momentum

	if not data.has_key('batch_size') or not type(data['batch_size']) is int:
		data['batch_size']=batch_size
	if not data.has_key('gbrbm_learning_rate') or not type(data['gbrbm_learning_rate']) is float:
		data['gbrbm_learning_rate'] = gbrbm_learning_rate
	if not data.has_key('learning_rate') or type(data['learning_rate']) is float:
		data['learning_rate'] = learning_rate
	if not data.has_key('pretraining_epochs') or not type(data['pretraining_epochs']) is int:
		data['pretraining_epochs'] = epochs
	if not data.has_key('keep_layer_num') or not type(data['keep_layer_num']) is int:
                data['keep_layer_num'] = keep_layer_num
	
	# momentum
	if data.has_key('initial_momentum') or not type(data['initial_momentum']) is float:
		data['initial_momentum']=initial_momentum
	if data.has_key('final_momentum') or not type(data['final_momentum']) is float:
		data['final_momentum']=final_momentum
	if data.has_key('initial_momentum_epoch ') or not type(data['initial_momentum_epoch']) is int:
		data['initial_momentum_epoch']=initial_momentum_epoch


	return data

#############################################################################################
#		SDA
#############################################################################################
def initModelSDA(data):

	finetune_lr=0.1
	pretraining_epochs=15
	pretrain_lr=0.08
	batch_size=1

	if not data.has_key('batch_size') or not type(data['batch_size']) is int:
		data['batch_size']=batch_size
	if not data.has_key('finetune_lr') or type(data['finetune_lr']) is float:
		data['finetune_lr'] = finetune_lr
	if not data.has_key('pretrain_lr') or type(data['pretrain_lr']) is float:
		data['pretrain_lr'] = pretrain_lr
	if not data.has_key('pretraining_epochs') or type(data['pretraining_epochs']) is int:
		data['pretraining_epochs'] = pretraining_epochs
	
	return data


def load_sda_spec(input_file):
	logger.info("Loading net properties from %s ..",input_file)	
	data = load_json(input_file)

	if not data.has_key('hidden_layers') or not type(data['hidden_layers']) is list:
		logger.critical(" hidden_layers is not present (or not a list) in " + str(input_file))
		exit(1)
	if not data.has_key('n_ins') or not type(data['n_ins']) is int:
		logger.critical(" n_ins is not present (or not a int) in " + str(input_file))
		exit(1)
	if not data.has_key('n_outs') or not type(data['n_outs']) is int:
		logger.critical(" n_outs is not present (or not a int) in " + str(input_file))
		exit(1)


	if not data.has_key('corruption_levels') or not type(data['corruption_levels']) is list:
		logger.critical(" corruption_levels is not present (or not a list) in " + str(input_file))
		exit(1)
	elif len(data['corruption_levels']) != len(data['hidden_layers']):
		logger.critical(" corruption_levels not correct size(should be same of hidden_layers) in " \
			+ str(input_file))
		exit(1)		

	if not data.has_key('random_seed') or not type(data['random_seed']) is int:
		data['random_seed'] = None


	return data




##############################################################################################
def __debugPrintData__(data,name=None):
	from json import dumps
	print name
	print dumps(data, indent=4, sort_keys=True)			

#if __name__ == '__main__':
#	#config_list = load_model(sys.argv[1])
#	config_list = load_conv_spec(sys.argv[1],256,[1,29,29,29])
#	print config_list;

