#!/usr/bin/env python2.7
# Copyright 2014	G.K SUDHARSHAN <sudharpun90@gmail.com>	IIT Madras
# Copyright 2014	Abil N George<mail@abilng.in>	IIT Madras
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



from utils.load_conf import load_model
import sys

import logging
logger = logging.getLogger(__name__)
from io_modules import setLogger


def setLoggerLevel(modelConfig,name=None):
	logger = logging.getLogger(name)
	# Set the level which determines what you see
	try:
		level = modelConfig['logger_level']
	except KeyError:
		level = "DEBUG"

	logger.info('Changing logger level:%s',level)

	if level == "INFO":
		logger.setLevel(logging.INFO)
	elif level == "DEBUG":
		logger.setLevel(logging.DEBUG)
	elif level == "ERROR":
		logger.setLevel(logging.ERROR)
	else:
		logger.setLevel(logging.WARNING)


def runNet(modelConfig):
	nnetType = modelConfig ['nnetType']
	logger.info("Loading Other Configuration for %s",nnetType);
	if nnetType == 'CNN':
		from run.run_CNN import runCNN as runModel
	elif nnetType == 'RBM':
		from run.run_DBN import runRBM as runModel
	elif nnetType == 'SDA':
		from run.run_SDA import runSdA as runModel
	elif nnetType == 'DNN':
		from run.run_DNN import runDNN as runModel
	else :
		logger.error('Unknown nnet Type')
		return 1
	runModel(modelConfig)


if __name__ == '__main__':
	setLogger();
	modelConfig = load_model(sys.argv[1])
	setLoggerLevel(modelConfig)
	runNet(modelConfig)
