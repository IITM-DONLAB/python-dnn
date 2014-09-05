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
	try:
		nnetType=data['nnetType']
	except KeyError, e:
		logger.critical("'nnetType' is missing in model properties file..")
		exit(1)

	if nnetType not in ['DNN','SDA','RBM','CNN']:
		logger.error('Unknown nnetType')
		exit(1)

	requiredKeys = ['data_spec','wdir','processes','nnet_spec','output_file','n_outs']
	if not isKeysPresents(data,requiredKeys):
		logger.critical(" the mandatory arguments are missing in model properties file..")
		exit(1)

	if not data.has_key('batch_size') or not type(data['batch_size']) is int:
		data['batch_size']=256

	if not data.has_key('random_seed') or not type(data['random_seed']) is int:
		data['random_seed'] = None
	
	if data.has_key('n_ins') or data.has_key('input_shape'):
		pass
	else:
		logger.error('Neither n_ins nor input_shape is present')
		logger.critical(" the mandatory arguments are missing in model properties file..")
		exit(1)

	specPaths=['data_spec','nnet_spec']
	data = correctPath(data,specPaths,input_file);

	wdirFiles=['output_file','conv_output_file','input_file',
                     'hidden_output_file','export_path','plot_path']
	data = correctPath(data,wdirFiles,data['wdir']+pathSep);

	#init Default Values in processes or update from Json.
	data['processes'] = initProcesses(data['processes'])

	#init Default Values of finetuning or update from Json.
	if data['processes']['finetuning']:
		data=initFinetuneParams(data);

	if data['processes']['pretraining']:
		data=initPreTrainParams(data);

	#__debugPrintData__(data,'model');
	return data;

def initFinetuneParams(data):
	if not data.has_key('finetune_params') or not type(data['finetune_params']) is dict:
		data['finetune_params'] = dict()
	finetune_params = data['finetune_params'];

	if not finetune_params.has_key('momentum') or not type(finetune_params['momentum']) is float:
		finetune_params['momentum']=0.5

	if not finetune_params.has_key('method'):
		finetune_params['method']="C"
	else:
		valid_methods = ['C','E']
	 	if finetune_params['method'] not in valid_methods:
			logger.error("Unknown finetuning method");
			logger.warning("Valid finetuning methods: "+str(valid_methods));
			exit(1);

	if finetune_params['method'] == "C":
		if (not finetune_params.has_key('learning_rate') or
			not type(finetune_params['learning_rate']) is float):
			finetune_params['learning_rate'] = 0.08
		if (not finetune_params.has_key('epoch_num') or
			not type(finetune_params['epoch_num']) is int):	
			finetune_params['epoch_num'] = 15
	else:
		if (not finetune_params.has_key('start_rate') or
			not type(finetune_params['start_rate']) is float):
			finetune_params['start_rate'] = 0.08
		if (not finetune_params.has_key('scale_by') or
			not type(finetune_params['learning_rate']) is float):
			finetune_params['scale_by'] = 0.08
		if (not finetune_params.has_key('min_derror_stop') or
			not type(finetune_params['min_derror_stop']) is float):
			finetune_params['min_derror_decay_start'] = 0.05
		if (not finetune_params.has_key('min_derror_stop') or 
			not type(finetune_params['min_derror_stop']) is float):
			finetune_params['min_derror_stop'] = 0.05
		if (not finetune_params.has_key('min_epoch_decay_start') or
			not type(finetune_params['min_epoch_decay_start']) is float):
			finetune_params['min_epoch_decay_start'] = 15
	return data

def initProcesses(data):
        if not data.has_key('pretraining') or not type(data['pretraining']) is bool:
                data['pretraining'] = False
        if not data.has_key("finetuning") or not type(data["finetuning"]) is bool:
                data["finetuning"] = False
        if not data.has_key('testing') or not type(data['testing']) is bool:
                data['testing'] = False
        if not data.has_key('export_data') or not type(data['export_data']) is bool:
                data['export_data'] = False
        if not data.has_key('plotting') or not type(data['plotting']) is bool:
                data['plotting'] = False
        return data

def initPreTrainParams(data):
	if not data.has_key('pretrain_params') or not type(data['pretrain_params']) is dict:
		data['pretrain_params'] = dict();
		logger.warning('Pretrain params not found.Using Default');

	if data['nnetType'] == 'RBM':
		data['pretrain_params'] = initPreTrainRBM(data['pretrain_params'])
	elif data['nnetType'] == 'SDA':
		data['pretrain_params'] = initPreTrainSDA(data['pretrain_params'])
	else:
		logger.warning("Pretraining of "+ data['nnetType'] + "is Not defined");
		data['processes']['pretraining'] = False;
	return data

def correctPath(data,keys,basePath):
	for key in keys:
		if data.has_key(key):
			data[key] = makeAbsolute(data[key],basePath)
	return data

def isKeysPresents(data,requiredKeys):
	for key in requiredKeys:
		if not data.has_key(key):
			logger.error('Missing Key in JSON :'+str(key))
			return False
	return True


def load_data_spec(input_file,batch_size):
	logger.info("Loading data specification properties from %s..",input_file)
	data = load_json(input_file);
	for x in ['training','testing','validation']:
		logger.debug('Validating data specification: %s',x)
		requiredKeys=['base_path','filename','partition','reader_type']

		if not data.has_key(x):
			continue;
		if not data[x].has_key('keep_flatten') or not type(data[x]['keep_flatten']) is bool:
			data[x]['keep_flatten'] = False

		if not data[x]['keep_flatten'] :
			requiredKeys.append('dim_shuffle');
		if not data[x].has_key('random') or not type(data[x]['keep_flatten']) is bool:
			data[x]['keep_flatten'] = True

		data[x]['batch_size'] = batch_size;

		if not isKeysPresents(data[x],requiredKeys):
			logger.critical("The mandatory arguments are missing in data spec(%s)",x)
			exit(1)

	return data


#############################################################################
#CNN
#############################################################################
def load_mlp_spec(spec):
	logger.info("Loading mlp properties from %s ...")
	if not spec.has_key('layers') or len(spec['layers'])==0:
		logger.critical("mlp configuration is not having layers key which is mandatory")
		exit(1);
					
	if not spec.has_key('do_dropout'):
		spec['do_dropout']=False;
		
	if spec['do_dropout'] and not spec.has_key('dropout_factor'):
		spec['dropout_factor']=0.2;
		
	if not spec.has_key('max_out'):
		spec['max_out']=None;
	else:
		if not spec['max_out'].has_key('method'):
			spec['max_out']['method'] = 'maxout'
		else:
			if not spec['max_out']['method'] in ['maxout','pnorm']:
				logger.critical("Invalid max_out method %s.."% spec['max_out']['method'])
				exit(1);
			
		if not spec['max_out'].has_key('pool_size'):
			spec['max_out']['pool_size'] = 1	
		
		if not spec['max_out'].has_key('pnorm_order'):
			spec['max_out']['pnorm_order'] = 1	
	if not spec['max_out'] is None and not spec['activation'] in ['linear','relu','cappedrelu']:
		spec['activation'] =  'linear';
		logger.debug("Setting the actiavtion function to linear, since max_out is used")
		 
	return spec;
		

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
		logger.critical("Error: No convnet configuration avaialable..")
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
		layer_configs[layer_index]['filter_shape'].extend(layer_configs[layer_index]['convmat_dim']);

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
	mlp_configs = load_mlp_spec(data['mlp']);
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


	if not data.has_key('pretrained_layers') or not type(data['pretrained_layers']) is int:
		data['pretrained_layers'] = len(data['hidden_layers'])
	elif data['pretrained_layers'] > (len(data['hidden_layers'])):
		data['pretrained_layers'] = len(data['hidden_layers'])

	first_layer_gb = True
	if data.has_key('first_layer_type') and data['first_layer_type'] == 'bb':
		first_layer_gb = False
	data['first_layer_gb'] = first_layer_gb

	#__debugPrintData__(data,'rbm');

	return (data)



def initPreTrainRBM(data):

	gbrbm_learning_rate = 0.005
	learning_rate = 0.08
	epochs=10
	keep_layer_num=0

	# momentum; more complicated than dnn
	initial_momentum = 0.5	 # initial momentum
	final_momentum = 0.9	   # final momentum
	initial_momentum_epoch = 5 # for how many epochs do we use initial_momentum

	if not data.has_key('keep_layer_num') or not type(data['keep_layer_num']) is int:
		data['keep_layer_num'] = keep_layer_num

	if not data.has_key('gbrbm_learning_rate') or not type(data['gbrbm_learning_rate']) is float:
		data['gbrbm_learning_rate'] = gbrbm_learning_rate
	if not data.has_key('learning_rate') or not type(data['learning_rate']) is float:
		data['learning_rate'] = learning_rate
	if not data.has_key('epochs') or not type(data['epochs']) is int:
		data['epochs'] = epochs

	# momentum
	if not data.has_key('initial_momentum') or not type(data['initial_momentum']) is float:
		data['initial_momentum']=initial_momentum
	if not data.has_key('final_momentum') or not type(data['final_momentum']) is float:
		data['final_momentum']=final_momentum
	if not data.has_key('initial_momentum_epoch ') or not type(data['initial_momentum_epoch']) is int:
		data['initial_momentum_epoch']=initial_momentum_epoch

	return data

#############################################################################################
#		SDA
#############################################################################################
def initPreTrainSDA(data):

	epochs=15
	learning_rate=0.08
	keep_layer_num = 0;

	if not data.has_key('keep_layer_num') or not type(data['keep_layer_num']) is int:
		data['keep_layer_num'] = keep_layer_num

	if not data.has_key('learning_rate') or not type(data['learning_rate']) is float:
		data['learning_rate'] = learning_rate
	if not data.has_key('epochs') or not type(data['epochs']) is int:
		data['epochs'] = epochs
	return data


def load_sda_spec(input_file):
	logger.info("Loading net properties from %s ..",input_file)
	data = load_json(input_file)

	if not data.has_key('hidden_layers') or not type(data['hidden_layers']) is list:
		logger.critical(" hidden_layers is not present (or not a list) in " + str(input_file))
		exit(1)

	if not data.has_key('corruption_levels') or not type(data['corruption_levels']) is list:
		logger.critical(" corruption_levels is not present (or not a list) in " + str(input_file))
		exit(1)
	elif len(data['corruption_levels']) != len(data['hidden_layers']):
		logger.critical(" corruption_levels not correct size(should be same of hidden_layers) in " \
			+ str(input_file))
		exit(1)


	return data

#############################################################################################
#		DNN
#############################################################################################
def load_dnn_spec(input_file):
	logger.info("Loading net properties from %s ..",input_file)
	data = load_json(input_file)

	if not data.has_key('hidden_layers') or not type(data['hidden_layers']) is list:
		logger.critical(" hidden_layers is not present (or not a list) in " + str(input_file))
		exit(1)

	if not data.has_key('pretrained_layers') or not type(data['pretrained_layers']) is int:
		data['pretrained_layers'] = -1;
	elif data['pretrained_layers'] > (len(data['hidden_layers'])):
		data['pretrained_layers'] = len(data['hidden_layers'])

	max_col_norm = None
	l1_reg = None
	l2_reg = None
	activation = "sigmoid"
	do_maxout = False
	pool_size = 1
	do_pnorm = False
	pnorm_order = 1
	do_dropout = False

	# regularization for hidden layer parameter
	if not data.has_key('max_col_norm') or not type(data['max_col_norm']) is float:
		data['max_col_norm'] = max_col_norm
	if not data.has_key('l1_reg') or not type(data['l1_reg']) is float:
		data['l1_reg'] = l1_reg
	if not data.has_key('l2_reg') or not type(data['l2_reg']) is float:
		data['l2_reg'] = l2_reg

	if not data.has_key('activation') :
		data['activation'] = 'sigmoid';
	if not data.has_key('do_maxout') or not type(data['do_maxout']) is bool:
		data['do_maxout'] = do_maxout
	if not data.has_key('pool_size') or not type(data['pool_size']) is int:
		data['pool_size'] = pool_size

	if not data.has_key('do_pnorm') or not type(data['do_pnorm']) is bool:
		data['do_pnorm'] = do_pnorm
		data['pnorm_order'] = pnorm_order
	if data['do_pnorm']:
		if not data.has_key('pnorm_order'):
			logger.critical("pnorm_order is not present in " + str(input_file))
			exit(1)			

	#dropout::
	if not data.has_key('do_dropout') or not type(data['do_dropout']) is bool:
		data['do_dropout'] = do_dropout
	if data['do_dropout']:
		if not data.has_key('dropout_factor') or not type(data['dropout_factor']) is list:
			logger.critical(" dropout_factor is not present (or not a list) in " + str(input_file))
			exit(1)
		elif len(data['dropout_factor']) != len(data['hidden_layers']):
			logger.critical(" dropout_factor not correct size(should be same of hidden_layers) in " \
				+ str(input_file))
			exit(1)
		if not data.has_key('input_dropout_factor') or not type(data['input_dropout_factor']) is float:
			logger.critical(" input_dropout_factor is not present (or not a list) in " + str(input_file))
			exit(1)


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

