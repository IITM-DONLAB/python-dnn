import json,sys


def load_json(input_file):
	with open(input_file) as data_file:
		data = json.load(data_file)  
	return data;

def load_model(input_file):
	print 'Loading model properties from ',input_file,' ...'

	data = load_json(input_file)  
	
	if (not data.has_key('data_spec')) or  (not data.has_key('conv_nnet_spec')) \
			or (not data.has_key('hidden_nnet_spec')) or (not data.has_key('wdir')) \
			or (not data.has_key('conv_output_file')) or (not data.has_key('hidden_output_file')):
		print "Error: the mandatory arguments are missing in model properties file.."
		exit(1)
	
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

	return data;

def load_data_spec(input_file):
	print 'Loading data specification properties from ',input_file,' ...'
	return load_json(input_file);

def load_mlp_spec(input_file):
	print 'Loading mlp properties from ',input_file,' ...'
	return load_json(input_file);

def load_conv_spec(input_file,batch_size,input_shape):
	print 'Loading convnet properties from ',input_file,' ...'	
	data = load_json(input_file)  
	
	layer_configs=data.pop('layers');
	conv_configs = data;
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
		layer_configs[layer_index]['filter_shape'].extend(layer_configs[layer_index]['convmat_dim']);
		
		layer_configs[layer_index]['output_shape'] = [batch_size,current_map_number];
		if not len(layer_configs[layer_index]['input_shape'][2:]) == len(layer_configs[layer_index]['convmat_dim']):
			print 'Input shape and convolution matrix dimension are not matching on layer ',layer_index+1
		input_shape=[current_map_number];
		for inp,wdim,pool in zip(layer_configs[layer_index]['input_shape'][2:],layer_configs[layer_index]['convmat_dim'],
				layer_configs[layer_index]['poolsize']):
			outdim = (inp-wdim+1)/pool
			layer_configs[layer_index]['output_shape'].append(outdim)
			input_shape.append(outdim);
	
		prev_map_number = current_map_number
	return (conv_configs,layer_configs)	
		

#if __name__ == '__main__':
#	#config_list = load_model(sys.argv[1])
#	config_list = load_conv_spec(sys.argv[1],256,[1,29,29,29])
#	print config_list;
	
