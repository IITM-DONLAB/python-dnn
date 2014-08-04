import json

def write_dataset(options):
	file_writer = FileWriter(options)
	file_writer.write_file_info()
	return file_writer


class FileWriter(object):
	def __init__(self, options):
		self.options = options
		self.filepath = options.pop('path')
		self.filehandle = open(self.filepath,'w+')

	def write_file_info(self):
    		self.filehandle.write(json.dumps(self.options)+'\n')

	def write_vector(self,vector_array,labels):
		fmt = self.options['format']
    		for vector,label in zip(vector_array,labels):
			flatten_vector = vector.flatten()
			if self.options['featdim']==len(flatten_vector):
				data={};data['x']=list(flatten_vector); data['y']=label;
		        	self.filehandle.write(json.dumps(data)+'\n')
			#else:
				#ignoring the feature vector
		
def read_dataset(options):
	file_reader = FileReader(options)
	file_header = file_reader.read_file_info()
	return file_reader,file_header

class FileReader(object):
	def __init__(self, options):
		self.options = options;
		self.filehandle = open(options['path'],'r');

	def read_file_info(self):
		jsonHeader= self.filehandle.readline();
		self.file_header = json.loads(jsonHeader);
		return self.file_header
	
	def read_pfile_data(self):
		while True:
			if self.frame_to_read == 0:
				break
	
'''
def read_data_args(data_spec):
	elements = data_spec.split(",")
	pfile_path = elements[0]
	dataset_args = {}
	for i in range(1, len(elements)):
		element = elements[i]
		arg_value = element.split("=")
		value = arg_value[1]
		key = arg_value[0]
		if key == 'partition':
			dataset_args['partition'] = 1024 * 1024 * int(value.replace('m',''))
		elif key == 'stream':
			dataset_args['stream'] = string_2_bool(value) # not supported for now
		elif key == 'random':
			dataset_args['random'] = string_2_bool(value)
		else:
			dataset_args[key] = int(value)  # left context & right context; maybe different
	return pfile_path, dataset_args
'''

'''
def read_dataset(pfile_path, read_opts):
	if read_opts['stream']:file
		pfile_reader = PfileDataReadStream(pfile_path, read_opts)
	else:
		pfile_reader = PfileDataRead(pfile_path, read_opts)
	# read pfile header and data
	pfile_reader.read_pfile_info()
	if (not read_opts['stream']):
		pfile_reader.read_pfile_data()
	
	shared_xy = pfile_reader.make_shared()
	shared_x, shared_y = shared_xy
	shared_y = T.cast(shared_y, 'int32')

	return pfile_reader, shared_xy, shared_x, shared_y

class FileReader(object):
	def read_pfile_info(self):
		pass
	def read_pfile_data(self):
		pass
	#load nth partition to the GPU shared memory
	def load_data_partition(self, n, shared_xy):
		pass
	def load_next_partition(self, shared_xy):
		pass
	def is_finish(self):
		pass
	def initialize_read(self):file
		pass
	def make_shared(self):
		pass

class StremFileReader(FileReader):

	def __init__(self, options):
		self.options = options;
		self.filehandle = open(options['path'])
		
	


		# pfile information
		self.header_size = 32768
		self.feat_start_column = 2
		self.feat_dim = 1024

		# markers while reading data
		self.frame_to_read = 0
		self.partition_num = 0
		self.frame_per_partition = 0
		
		# store number of frames, features and labels for each data partition
		self.frame_nums = []
		self.feat_mats = []
		self.label_vecs = []

		# other variables to be consistent with PfileDataReadStream
		self.partition_index = 0
		self.cur_frame_num = 0
		self.end_reading = False

	# read pfile information from the header part
	def read_pfile_info(self):
		line = self.file_read.readline()
		if line.startswith('-pfile_header') == False:
			print "Error: PFile format is wrong, maybe the file was not generated successfully."
			exit(1)
		self.header_size = int(line.split(' ')[-1])
		while (not line.startswith('-end')):
			if line.startswith('-num_frames'):
				self.frame_to_read = int(line.split(' ')[-1])
			elif line.startswith('-first_feature_column'):
				self.feat_start_column = int(line.split(' ')[-1])
			elif line.startswith('-num_features'):
				self.feat_dim = int(line.split(' ')[-1])
			line = self.file_read.readline()
		# partition size in terms of frames
		self.frame_per_partition = self.read_opts['partition'] / (self.feat_dim * 4)
		batch_residual = self.frame_per_partition % 256
		self.frame_per_partition = self.frame_per_partition - batch_residual
		
	def read_pfile_data(self):
		# data format for pfile reading
		# s -- sentence index; f -- frame index; d -- features; l -- label
		self.dtype = numpy.dtype({'names': ['d', 'l'],
								'formats': [('>f', self.feat_dim), '>i'],
								'offsets': [self.feat_start_column * 4, (self.feat_start_column + self.feat_dim) * 4]})
		# Now we skip the file header
		self.file_read.seek(self.header_size, 0)
		while True:
			if self.frame_to_read == 0:
				break
			frameNum_this_partition = min(self.frame_to_read, self.frame_per_partition)
			partition_array = numpy.fromfile(self.file_read, self.dtype, frameNum_this_partition)
			feat_mat = numpy.asarray(partition_array['d'], dtype = theano.config.floatX)
			label_vec = numpy.asarray(partition_array['l'], dtype = theano.config.floatX)
			self.feat_mats.append(feat_mat)
			self.label_vecs.append(label_vec)
			self.frame_nums.append(len(label_vec))
			self.frame_to_read = self.frame_to_read - frameNum_this_partition
			self.partition_num = self.partition_num + 1
		# finish reading; close the file
		self.file_read.close()
  
	# load the n-th (0 indexed) partition to the GPU memory 
	def load_data_partition(self, n, shared_xy):
		shared_x, shared_y = shared_xy
		feat = self.feat_mats[n]
		label = self.label_vecs[n]

		if self.read_opts['random']:  # randomly shuffle features and labels in the *same* order
			numpy.random.seed(18877)
			numpy.random.shuffle(feat)
			numpy.random.seed(18877)
			numpy.random.shuffle(label)

		shared_x.set_value(feat, borrow=True)
		shared_y.set_value(label, borrow=True)
 
	def load_next_partition(self, shared_xy):
		self.load_data_partition(self.partition_index, shared_xy)
		self.cur_frame_num = self.frame_nums[self.partition_index]
		self.partition_index = self.partition_index + 1
		if self.partition_index >= self.partition_num:
			self.end_reading = True
			self.partition_index = 0

	def is_finish(self):
		return self.end_reading

	# reopen pfile with the same filename
	def reopen_file(self):
		self.file_read = open(self.pfile_path)
		self.read_pfile_info()
		self.initialize_read()
		self.read_pfile_data()

	def initialize_read(self):
		self.end_reading = False
		self.partition_index = 0

	def make_shared(self):
		# define shared variables
		feat = self.feat_mats[0]
		label = self.label_vecs[0]
		
		if self.read_opts['random']:  # randomly shuffle features and labels in the *same* order
			numpy.random.seed(18877)
			numpy.random.shuffle(feat)
			numpy.random.seed(18877)
			numpy.random.shuffle(label)

		shared_x = theano.shared(feat, name = 'x', borrow = True)
		shared_y = theano.shared(label, name = 'y', borrow = True)
		return shared_x, shared_y

class PfileDataReadStream(object):

	def __init__(self, pfile_path, read_opts):

		self.file_read = open(pfile_path)
		self.pfile_path = pfile_path
		self.read_opts = read_opts

		# pfile information
		self.header_size = 32768
		self.feat_start_column = 2
		self.feat_dim = 1024
		self.total_frame_num = 0

		# markers while reading data
		self.frame_to_read = 0
		self.partition_num = 0
		self.frame_per_partition = 0

		# store number of frames, features and labels for each data partition
		self.feat = None
		self.label = None
		self.cur_frame_num = 0
		self.end_reading = False

	# read pfile information from the header part
	def read_pfile_info(self):
		line = self.file_read.readline()
		if line.startswith('-pfile_header') == False:
			print "Error: PFile format is wrong, maybe the file was not generated successfully."
			exit(1)
		self.header_size = int(line.split(' ')[-1])
		while (not line.startswith('-end')):
			if line.startswith('-num_frames'):
				self.total_frame_num = self.frame_to_read = int(line.split(' ')[-1])
			elif line.startswith('-first_feature_column'):
				self.feat_start_column = int(line.split(' ')[-1])
			elif line.startswith('-num_features'):
				self.feat_dim = int(line.split(' ')[-1])
			line = self.file_read.readline()
		# partition size in terms of frames
		self.frame_per_partition = self.read_opts['partition'] / (self.feat_dim * 4)
		batch_residual = self.frame_per_partition % 256
		self.frame_per_partition = self.frame_per_partition - batch_residual
		
	def read_one_partition(self):
		# data format for pfile reading
		# d -- features; l -- label
		self.dtype = numpy.dtype({'names': ['d', 'l'],
								'formats': [('>f', self.feat_dim), '>i'],
								'offsets': [self.feat_start_column * 4, (self.feat_start_column + self.feat_dim) * 4]})
		if self.feat is None:  # haven't read anything, then skip the file header
			self.file_read.seek(self.header_size, 0)
		
		frameNum_this_partition = min(self.frame_to_read, self.frame_per_partition)
		partition_array = numpy.fromfile(self.file_read, self.dtype, frameNum_this_partition)
		self.feat = numpy.asarray(partition_array['d'], dtype = theano.config.floatX)
		self.label = numpy.asarray(partition_array['l'], dtype = theano.config.floatX)
		self.cur_frame_num = frameNum_this_partition
		self.frame_to_read = self.frame_to_read - frameNum_this_partition
		if self.frame_to_read <= 0:
			self.end_reading = True
			self.file_read.seek(self.header_size, 0)
			self.frame_to_read = self.total_frame_num

	# reopen pfile with the same filename
	def reopen_file(self):
		self.file_read = open(self.pfile_path)
		self.read_pfile_info()
		self.initialize_read() 

	def is_finish(self):
		return self.end_reading

	def initialize_read(self):
		self.end_reading = False
		self.file_read.seek(self.header_size, 0)
		self.frame_to_read = self.total_frame_num

	# load the n-th (0 indexed) partition to the GPU memory 
	def load_next_partition(self, shared_xy):
		shared_x, shared_y = shared_xy
		
		# read one partition from disk
		self.read_one_partition()
		if self.read_opts['random']:  # randomly shuffle features and labels in the *same* order
			numpy.random.seed(18877)
			numpy.random.shuffle(self.feat)
			numpy.random.seed(18877)
			numpy.random.shuffle(self.label)
			
		shared_x.set_value(self.feat, borrow=True)
		shared_y.set_value(self.label, borrow=True)

	def make_shared(self):
		if self.feat is None:
			self.read_one_partition()
			self.initialize_read()

		if self.read_opts['random']:  # randomly shuffle features and labels in the *same* order
			numpy.random.seed(18877)
			numpy.random.shuffle(self.feat)
			numpy.random.seed(18877)
			numpy.random.shuffle(self.label)

		shared_x = theano.shared(self.feat, name = 'x', borrow = True)
		shared_y = theano.shared(self.label, name = 'y', borrow = True)
		return shared_x, shared_y

'''
