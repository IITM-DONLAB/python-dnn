#!/usr/bin/env python2.7
# Copyright 2014	G.K SUDHARSHAN <sudharpun90@gmail.com>    IIT Madras
# Copyright 2014	Abil N George<mail@abilng.in>    IIT Madras
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

import cPickle, gzip, os, sys, time, numpy, json

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from io_modules.model_io import _nnet2file, _file2nnet, _cnn2file, _file2cnn
from utils.load_conf import load_model, load_conv_spec,load_mlp_spec,load_data_spec 
from io_modules.data_exporter import export_data


from models.cnn import CNN
from utils.utils import parse_activation

def runCNNFeat(configFile):
	model_configs = load_model(configFile,'CNN')
	batch_size = model_configs['batch_size'];
	conv_configs,conv_layer_configs = load_conv_spec(model_configs['conv_nnet_spec'],model_configs['batch_size'],
				model_configs['input_shape'])
	mlp_configs = load_mlp_spec(model_configs['hidden_nnet_spec']);
	conv_activation = parse_activation(conv_configs['activation']);
	hidden_activation = parse_activation(mlp_configs['activation']);

	numpy_rng = numpy.random.RandomState(89677)
	theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

	print('> ... loading the model')
	cnn = CNN(numpy_rng,theano_rng,conv_layer_configs = conv_layer_configs, batch_size = batch_size,
		n_outs=model_configs['n_outs'],hidden_layers_sizes=mlp_configs['layers'], conv_activation = conv_activation,
		hidden_activation = hidden_activation,use_fast = conv_configs['use_fast'])

	
	_file2cnn(cnn.conv_layers, filename=model_configs['conv_output_file'], activation=conv_activation)
	out_fn = cnn.build_out_function();
	
	data_spec = load_data_spec(model_configs['data_spec'])
	export_data(data_spec['testing'],model_configs['export_path'],out_fn,cnn.conv_output_dim);
	
#######################################################################################################################################################
'''
    arg_elements=[]
    for i in range(1, len(sys.argv)):
        arg_elements.append(sys.argv[i])
    arguments = parse_arguments(arg_elements)

    if (not arguments.has_key('ark_file')) or (not arguments.has_key('conv_layer_number')) or (not arguments.has_key('wdir')) or (not arguments.has_key('output_file_prefix')) or (not arguments.has_key('conv_net_file')):
        print "Error: the mandatory arguments are: --ark-file --conv-layer-num --conv-input-file --wdir --output-file-prefix"
        exit(1)

    # mandatory arguments
    ark_file = arguments['ark_file']
    conv_layer_number = int(arguments['conv_layer_number'])
    wdir = arguments['wdir']
    output_file_prefix = arguments['output_file_prefix']
    conv_net_file = arguments['conv_net_file']
    # network structure
    conv_configs = []
    for i in xrange(conv_layer_number):
      config_path = wdir + '/conv.config.' + str(i)
      if os.path.exists(config_path) == False:
          print "Error: config files for convolution layers do not exist."
          exit(1)
      else:
          with open(config_path, 'rb') as fp:
              conv_configs.append(json.load(fp))
          # convert string activaton to theano
          conv_configs[i]['activation'] = parse_activation(conv_configs[i]['activation'])

    # whether to use the fast mode
    use_fast = False
    if arguments.has_key('use_fast'):
        use_fast = string_2_bool(arguments['use_fast'])

    # paths for output files
    output_scp = output_file_prefix + '.scp'
    output_ark = output_file_prefix + '.ark'

    start_time = time.clock()
    feat_rows = []
    feat_mats_np = []
    uttIDs = []

    kaldiIn = KaldiReadIn(ark_file)
    kaldiIn.open()
    uttID, feat_mat = kaldiIn.next()
    while 1:
        num_row, num_col = feat_mat.shape
        feat_rows.append(num_row)
        feat_mats_np.append(feat_mat)
        uttIDs.append(uttID)
        uttID, feat_mat = kaldiIn.next()
        if feat_mat == None:
            break
    kaldiIn.close()

    input_shape_train = conv_configs[0]['input_shape']
    input_shape_1 = (input_shape_train[1], input_shape_train[2], input_shape_train[3])
    num_utt = len(feat_mats_np)
    feat_mats = []
    for i in xrange(num_utt):
        feat_mats.append(numpy.reshape(feat_mats_np[i], (feat_rows[i],) + input_shape_1))

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    cnn = CNN_Forward(numpy_rng = rng, theano_rng=theano_rng,
                 conv_layer_configs = conv_configs, use_fast = use_fast)
    _file2cnn(cnn.conv_layers, filename=conv_net_file)
    out_function = cnn.build_out_function(feat_mats)

    log('> ... processing the data')

    kaldiOut = KaldiWriteOut(output_scp,output_ark)
    kaldiOut.open()
    for i in xrange(num_utt):
        feat_out = out_function(feat_mats[i])
        kaldiOut.write(uttIDs[i], feat_out)
    kaldiOut.close()

    end_time = time.clock()
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
'''
if __name__ == '__main__':
	runCNNFeat(sys.argv[1])

