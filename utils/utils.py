import theano.tensor as T
from learn_rates import LearningRateConstant, LearningRateExpDecay

def string_2_bool(string):
    if string == 'true':
        return True
    if string == 'false':
        return False

def parse_arguments(arg_elements):
    args = {}
    arg_num = len(arg_elements) / 2
    for i in xrange(arg_num):
        key = arg_elements[2*i].replace("--","").replace("-", "_");
        args[key] = arg_elements[2*i+1]
    return args

def dimshuffle(a,shuffle):
	len = shuffle.__len__();
	ishuffle = [0]*len;
	for index,dim in zip(xrange(len),shuffle):
		ishuffle[dim]=index
	shuffle = ishuffle;
	index=0;
	while(index<len):
		if index==shuffle[index]:
			index=index+1;
		else:
			swapwith = shuffle[index];
			a=a.swapaxes(index,swapwith);
			shuffle[index],shuffle[swapwith]=shuffle[swapwith],shuffle[index]
	return a

'''
#older code
def parse_lrate(lrate_string):
    elements = lrate_string.split(":")
    # 'D:0.08:0.5:0.05,0.05:15'
    if elements[0] == 'D':  # ExpDecay
        if (len(elements) != 5):
            return None
        values = elements[3].split(',')
        lrate = LearningRateExpDecay(start_rate=float(elements[1]),
                                 scale_by = float(elements[2]),
                                 min_derror_decay_start = float(values[0]),
                                 min_derror_stop = float(values[1]),
                                 init_error = 100,
                                 min_epoch_decay_start=int(elements[4]))
        return lrate

    # 'C:0.08:15'
    if elements[0] == 'C':  # Constant
        if (len(elements) != 3):
            return None
        lrate = LearningRateConstant(learning_rate=float(elements[1]),
                                 epoch_num = int(elements[2]))
        return lrate

def parse_conv_spec(conv_spec, batch_size):
    # "1x29x29:100,5x5,p2x2:200,4x4,p2x2,f"
    conv_spec = conv_spec.replace('X', 'x')
    structure = conv_spec.split(':')
    conv_layer_configs = []
    for i in range(1, len(structure)):
        config = {}
        elements = structure[i].split(',')
        if i == 1:
            input_dims = structure[i - 1].split('x')
            prev_map_number = int(input_dims[0])
            prev_feat_dim_x = int(input_dims[1])
            prev_feat_dim_y = int(input_dims[2])
        else:
            prev_map_number = conv_layer_configs[-1]['output_shape'][1]
            prev_feat_dim_x = conv_layer_configs[-1]['output_shape'][2]
            prev_feat_dim_y = conv_layer_configs[-1]['output_shape'][3]

        current_map_number = int(elements[0])
        filter_xy = elements[1].split('x')
        filter_size_x = int(filter_xy[0])
        filter_size_y = int(filter_xy[1])
        pool_xy = elements[2].replace('p','').replace('P','').split('x')
        pool_size_x = int(pool_xy[0])
        pool_size_y = int(pool_xy[1])
        output_dim_x = (prev_feat_dim_x - filter_size_x + 1) / pool_size_x
        output_dim_y = (prev_feat_dim_y - filter_size_y + 1) / pool_size_y

        config['input_shape'] = (batch_size, prev_map_number, prev_feat_dim_x, prev_feat_dim_y)
        config['filter_shape'] = (current_map_number, prev_map_number, filter_size_x, filter_size_y)
        config['poolsize'] = (pool_size_x, pool_size_y)
        config['output_shape'] = (batch_size, current_map_number, output_dim_x, output_dim_y)
        if len(elements) == 4 and elements[3] == 'f':
            config['flatten'] = True
        else:
            config['flatten'] = False

        conv_layer_configs.append(config)
    return conv_layer_configs
'''
 
def parse_activation(act_str):
    if act_str == 'sigmoid':
        return T.nnet.sigmoid
    if act_str == 'tanh':
        return T.tanh
    return T.nnet.sigmoid

def activation_to_txt(act_func):
    if act_func == T.nnet.sigmoid:
        return 'sigmoid'
    if act_func == T.tanh:
        return 'tanh'

def parse_two_integers(argument_str):
    elements = argument_str.split(":")
    int_strs = elements[1].split(",")
    return int(int_strs[0]), int(int_strs[1])


