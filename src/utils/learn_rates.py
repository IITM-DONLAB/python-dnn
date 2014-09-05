class LearningRate(object):
	@staticmethod
	def get_instance(configs):
		method = configs['method'];
		if(method=='C'):
			return LearningRateConstant(configs);
		else:
			return LearningRateExpDecay(configs);
	def __init__(self):
		'''constructor'''	
		
	def get_rate(self):
		pass

	def get_next_rate(self, current_error):
		pass

class LearningRateConstant(LearningRate):
	#learning_rate = 0.08, epoch_num = 20
	def __init__(self, kwargs):

		self.learning_rate = kwargs['learning_rate']
		self.epoch = 1
		self.epoch_num = kwargs['epoch_num']
		self.rate = self.learning_rate

	def get_rate(self):
		return self.rate

	def get_next_rate(self, current_error):

		if ( self.epoch >=  self.epoch_num):
			self.rate = 0.0
		else:
			self.rate = self.learning_rate
		self.epoch += 1

		return self.rate

class LearningRateExpDecay(LearningRate):
	#start_rate = 0.08, scale_by = 0.5,min_derror_decay_start = 0.05, min_derror_stop = 0.05, init_error = 100, min_epoch_decay_start=15, 
	def __init__(self,kwargs, decay=False, zero_rate = 0.0):

		self.start_rate = kwargs['start_rate']
		self.rate = kwargs['start_rate']
		self.scale_by = kwargs['scale_by']
		self.min_derror_decay_start = kwargs['min_derror_decay_start']
		self.min_derror_stop = kwargs['min_derror_stop']
		self.min_epoch_decay_start = kwargs['min_epoch_decay_start']
		self.lowest_error = float('Inf')
		self.epoch = 1
		self.decay = decay
		self.zero_rate = zero_rate


	def get_rate(self):
		return self.rate  
	
	def get_next_rate(self, current_error):
		diff_error = 0.0
		diff_error = self.lowest_error - current_error
			
		if (current_error < self.lowest_error):
			self.lowest_error = current_error
	
		if (self.decay):
			if (diff_error < self.min_derror_stop):
				self.rate = 0.0
			else:
				self.rate *= self.scale_by
		else:
			if ((diff_error < self.min_derror_decay_start) and (self.epoch > self.min_epoch_decay_start)):
				self.decay = True
				self.rate *= self.scale_by
			
		self.epoch += 1
		return self.rate


class LearningMinLrate(LearningRate):

	def __init__(self, start_rate = 0.08, scale_by = 0.5,
				 min_lrate_stop = 0.0002, init_error = 100,
				 decay=False, min_epoch_decay_start=15):

		self.start_rate = start_rate
		self.init_error = init_error

		self.rate = start_rate
		self.scale_by = scale_by
		self.max_epochs = max_epochs
		self.min_lrate_stop = min_lrate_stop
		self.lowest_error = init_error

		self.epoch = 1
		self.decay = decay
		self.min_epoch_decay_start = min_epoch_decay_start

	def get_rate(self):
		return self.rate

	def get_next_rate(self, current_error):
		diff_error = 0.0

		diff_error = self.lowest_error - current_error

		if (current_error < self.lowest_error):
			self.lowest_error = current_error

		if (self.decay):
			if (self.rate < self.min_lrate_stop):
				self.rate = 0.0
			else:
				self.rate *= self.scale_by
		else:
			if (self.epoch >= self.min_epoch_decay_start):
				self.decay = True
				self.rate *= self.scale_by

		self.epoch += 1
		return self.rate
