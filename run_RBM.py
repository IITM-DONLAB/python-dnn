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



from utils.load_conf import load_model,load_rbm_spec,load_data_spec

def runRBM(configFile):
	model_config = load_model(configFile)

	rbm_config,rbmlayer_config = load_rbm_spec(model_config['rbm_nnet_spec'],model_config['batch_size'],
				model_config['input_shape'])
	#mlp_config = load_mlp_spec(model_config['hidden_nnet_spec']);
	data_spec =  load_data_spec(model_config['data_spec']);





if __name__ == '__main__':
    import sys
    runRBM(sys.argv[1])
