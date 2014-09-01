import theano.tensor as T

from os.path import isabs,dirname,abspath,join
from os.path import sep as pathSep


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


def parse_activation(activation):
    if activation == 'tanh':
        return T.tanh
    elif activation == 'sigmoid':
        return T.nnet.sigmoid
    elif activation == 'relu':
        return lambda x: x * (x > 0)
    elif activation == 'cappedrelu':
        return lambda x: T.minimum(x * (x > 0), 6)
    else:
        raise NotImplementedError

def activation_to_txt(act_func):
    if act_func == T.nnet.sigmoid:
        return 'sigmoid'
    if act_func == T.tanh:
        return 'tanh'

def parse_two_integers(argument_str):
    elements = argument_str.split(":")
    int_strs = elements[1].split(",")
    return int(int_strs[0]), int(int_strs[1])



def makeAbsolute(path,base='/'):
    if isabs(path) :
        return path
    else :
        if not base[-1] == pathSep : 
            base = dirname(base);
        base = abspath(base)
        path = join(base,path);
        return path