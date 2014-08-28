import json,numpy,sys,os
import theano
import theano.tensor as T
from utils.utils import dimshuffle

import logging
logger = logging.getLogger(__name__)

def read_dataset(options,pad_zeros=False):
	filepath =   options['base_path'] + os.sep + options['filename'];
	logger.info("%s dataset will be initialized to reader to %s",
				options['reader_type'],filepath);
	logger.debug("options : %s" % str(options))	
				
	file_reader = FileReader.get_instance(filepath,options)
	file_header = file_reader.read_file_info()

	shared_xy = file_reader.create_shared(pad_zeros)
	shared_x, shared_y = shared_xy
	shared_y = T.cast(shared_y, 'int32')
	return file_reader,shared_xy, shared_x, shared_y

##########################################BASE CLASS##############################################################

class FileReader(object):
	'''Gets instance of file based on the reader type'''
	filepath = None;
	options=None;
	filehandle = None
	# store features and labels for each data partition
	feat = None
	label = None
	# markers while reading data
	partition_num = 0
	frame_per_partition = 0
	end_reading = False
	feat_dim = 0;
	cur_frame_num = 0;
	num_pad_frames = 0;
	
	@staticmethod
	def get_instance(filepath,options):
		file_reader = None;
		if options['reader_type']=='NP':
			file_reader = NPFileReader(filepath,options);
		elif options['reader_type']=='TD':
			file_reader = TDFileReader(filepath,options);
		elif options['reader_type']=='T1':
			file_reader = T1FileReader(filepath,options);
		elif options['reader_type']=='T2':
			file_reader = T2FileReader(filepath,options);
		else:
			logger.critical('\'%s\'  reader_type is not defined...'\
						%options['reader_type'])
		return file_reader
		
	'''Reads the file header information'''	
	def read_file_info(self):
		pass
		
	'''Reads the data from the next partition'''	
	def read_next_partition_data(self,already_read=0,pad_zeros=False):
		pass
		
	'''Makes the current partition shared across GPU's'''	
	def make_partition_shared(self, shared_xy):
		logger.debug("Partition is now made shared for GPU processing");
		shared_x, shared_y = shared_xy  
		if self.options['random']:  # randomly shuffle features and labels in the *same* order
			try: 
				seed = self.options['random_seed']
			except KeyError:
				seed = 18877
			numpy.random.seed(seed)
			numpy.random.shuffle(self.feat)	
			numpy.random.seed(seed)
			numpy.random.shuffle(self.label)
		shared_x.set_value(self.feat, borrow=True)
		shared_y.set_value(self.label, borrow=True)
		
		
	def pad_zeros(self,num_pad_frames):
		if num_pad_frames > 0:
			logger.debug("Padded %d frames for one partition" % num_pad_frames);
			self.num_pad_frames = num_pad_frames;
			for x in xrange(self.num_pad_frames):
				self.feat = numpy.append(self.feat,[0]*self.feat_dim) 
				self.cur_frame_num+=1
				
				
	'''Create first partition shared across GPU's'''	
	def create_shared(self,pad_zeros=False):
		if self.feat is None:
			self.read_next_partition_data(pad_zeros)
		shared_x = theano.shared(self.feat, name = 'x', borrow = True)
		shared_y = theano.shared(self.label, name = 'y', borrow = True)
		return shared_x, shared_y

	'''Checks if the file reading reached end of file '''
	def is_finish(self):
		return self.end_reading
		
	''' Initialize the file_reader options''' 	
	def initialize_read(self):
		logger.debug("File reader is initialzed again for file reading");	
		self.filehandle.seek(0,0);
		self.end_reading = False
		self.feat=None
		self.labels=None
		self.partition_num = 0
		self.frames_per_partition= 0	
		self.num_pad_frames = 0
		self.read_file_info()
		self.read_next_partition_data()		

##########################################TD FILEREADER##############################################################
"""
	Reads the simple text file following is the structure
	<feat_dim> <num_feat_vectors>(optional)
	<feat_vector>
	<feat_vector>
	.
	.
"""

class TDFileReader(FileReader):
	''' Reads the data stored in as Simple Text File'''
	def __init__(self,path,options):
		self.filepath = path;
		self.options = options;
		self.batch_size = options['batch_size']
		self.lbl = options['label'];
		self.filehandle = open(self.filepath,'rb')
		
	def read_file_info(self):
		header = self.filehandle.readline();
		self.header = header.split()
		self.feat_dim = int(self.header[0]);
		self.header = {}	
		self.header['featdim'] = self.feat_dim
		logger.debug('TD Filereader : feats : %d, label : %d' % (self.feat_dim,self.lbl))
		#self.frames_remaining = long(self.header[1])
		# partitions specifies approximate amount data to be loaded one operation
		self.frames_per_partition= self.options['partition'] *1000*1000/ (self.feat_dim * 4)
		batch_residual = self.frames_per_partition% self.batch_size
		self.frames_per_partition = self.frames_per_partition - batch_residual
		return self.header
		
	def read_next_partition_data(self,already_read=0,pad_zeros=False):
		self.feat=[]
		self.cur_frame_num = 0
		while  self.cur_frame_num < self.frames_per_partition-already_read:
			values = self.filehandle.readline().split()
			if values.__len__()==0: #No more values available in the data file
				break;
			fvalues = [float(value) for value in values];
			self.feat = numpy.append(self.feat,fvalues)
			self.cur_frame_num += 1
		
		if self.cur_frame_num  >  0:
			if pad_zeros:
				self.pad_zeros(int(self.frames_per_partition)-already_read-self.cur_frame_num);
			self.feat = self.feat.reshape([self.cur_frame_num,self.feat_dim])
			self.feat = numpy.asarray(self.feat, dtype = theano.config.floatX)
			self.label = numpy.asarray([self.lbl]*self.cur_frame_num , dtype = theano.config.floatX)
			self.partition_num = self.partition_num + 1
			logger.debug('TD Filereader : %d frames read from %s' % (self.cur_frame_num,self.filepath))
			if not self.options['keep_flatten'] :	#reshape the vector if needed
				logger.debug('TD Filereader : Reshape input...')
				shape = [self.cur_frame_num];
				shape.extend(self.options['input_shape']);
				self.feat = self.feat.reshape(shape);
				self.feat = dimshuffle(self.feat,self.options['dim_shuffle'])				
		else:
			self.end_reading = True;
		
##########################################T2 FILEREADER##############################################################
"""
	Reads the dataset which has two level directory structure
	Root data file...................................................................................................
	<feat_dim> <num_classes>
	<class_index_file1>
	<class_index_file2>
	.
	.
	Class index file................................................................................................
	<td_data_file1>
	<td_data_file2>
	.
	.
"""			

class T2FileReader(FileReader):
	''' Reads the data stored in as Simple Text File With Two level header structure'''
	def __init__(self,path,options):
		self.filepath = path;
		self.options=options;
		self.batch_size = options['batch_size']
		self.filehandle = open(self.filepath,'rb')
		
	def read_file_info(self):
		# initializes multiple class handles 
		self.filehandles = [];
		self.childhandles=[];
		self.last_class_idx = 0;
		
		header = self.filehandle.readline();
		self.header = header.split()
		self.feat_dim = int(self.header[0]);
		self.classes = long(self.header[1])
		
		logger.debug('T2 Filereader : feat : %d' % self.feat_dim)
		
		self.header = {}
		self.header['featdim'] = self.feat_dim
		self.header['classes'] = self.classes
		
		batch_size = self.batch_size
		
		# partitions specifies approximate amount data to be loaded one operation
		self.frames_per_partition = self.options['partition'] *1000*1000/ (self.feat_dim * 4)
		batch_residual = self.frames_per_partition% batch_size
		self.frames_per_partition= self.frames_per_partition- batch_residual
		
		#load filehandle for all classes
		for i in xrange(0,self.classes):
			data_file = self.options['base_path'] + os.sep + self.filehandle.readline().strip();
			self.filehandles.append(open(data_file,'rb'));
			self.childhandles.append(None);
			
		if self.frames_per_partition < self.classes:
			logger.critical('Number of frames per partition must be greater than the number of classes, \
				Please increase the partition size');
		self.frames_per_class = self.frames_per_partition/self.classes
		
		return self.header
		
	def read_next_partition_data(self,already_read=0,pad_zeros=False):
		self.cur_frame_num = 0
		self.feat = []
		self.label = []
		none_cnt = 0
		
		while self.cur_frame_num < self.frames_per_partition and none_cnt < self.classes :
			if self.childhandles[self.last_class_idx] is None:	#if the child handle is not initialized
				data_file = self.filehandles[self.last_class_idx].readline().strip();
				if data_file.__len__() != 0:
					child_options = self.options.copy()
					child_options['filename']=data_file
					child_options['label']= self.last_class_idx
					child_options['keep_flatten'] = True
					data_file = child_options['base_path'] + os.sep + child_options['filename']
					self.childhandles[self.last_class_idx] = TDFileReader(data_file,child_options)
					self.childhandles[self.last_class_idx].read_file_info()
					
			if not self.childhandles[self.last_class_idx] is None: 
				none_cnt = 0
				self.childhandles[self.last_class_idx].read_next_partition_data(already_read=self.frames_per_partition-self.frames_per_class)
				if self.childhandles[self.last_class_idx].cur_frame_num > 0:
					self.feat = numpy.append(self.feat,self.childhandles[self.last_class_idx].feat)
					self.label = numpy.append(self.label,self.childhandles[self.last_class_idx].label)
					self.cur_frame_num += self.childhandles[self.last_class_idx].cur_frame_num

				if self.childhandles[self.last_class_idx].is_finish():
					self.childhandles[self.last_class_idx] = None
			else:
				none_cnt+=1
	
			self.last_class_idx = (self.last_class_idx + 1)	% self.classes
			
		if self.cur_frame_num >  0:
			if pad_zeros:	#padding zeros
				self.pad_zeros(int(self.frames_per_partition)-already_read-self.cur_frame_num);
				
			logger.debug('T2 Filereader : from file %s, %d partition has %d frames' % (self.filepath,self.partition_num,self.cur_frame_num));
			self.feat = self.feat.reshape([self.cur_frame_num,self.feat_dim])
			self.feat = numpy.asarray(self.feat, dtype = theano.config.floatX)
			self.partition_num = self.partition_num + 1
			
			if not self.options['keep_flatten'] :	#reshape the vector if needed
				logger.debug('T2 Filereader : Reshape input...')
				shape = [self.cur_frame_num];
				shape.extend(self.options['input_shape']);
				self.feat = self.feat.reshape(shape);
				self.feat = dimshuffle(self.feat,self.options['dim_shuffle'])
		else:
			self.end_reading = True
		

##########################################T1 FILEREADER##############################################################
"""
	Reads the dataset which has single level directory structure
	Root data file...................................................................................................
	<feat_dim> <num_classes>
	<td_data_file1>
	<td_data_file1>
	.
	.
"""			

class T1FileReader(FileReader):
	''' Reads the data stored in as Simple Text File With One level header structure'''
	def __init__(self,path,options):
		self.filepath = path;
		self.options=options;
		self.batch_size = options['batch_size']
		self.filehandle = open(self.filepath,'rb')
		
	def read_file_info(self):
		# initializes multiple class handles 
		self.filehandles = [];
		self.last_class_idx = 0;
		
		header = self.filehandle.readline();
		self.header = header.split()
		self.feat_dim = int(self.header[0]);
		self.classes = long(self.header[1])
		
		self.header = {};
		self.header['featdim'] = self.feat_dim
		batch_size = self.batch_size
		
		logger.debug('T1 Filereader : feat : %d' % self.feat_dim)
		
		# partitions specifies approximate amount data to be loaded one operation
		self.frames_per_partition = self.options['partition'] *1000*1000/ (self.feat_dim * 4)
		batch_residual = self.frames_per_partition% batch_size
		self.frames_per_partition= self.frames_per_partition- batch_residual
		
		#load filehandle for all classes
		for i in xrange(0,self.classes):
			child_options = self.options.copy();
			child_options['filename'] = self.filehandle.readline().strip();
			child_options['label']= i;
			child_options['keep_flatten'] = True
			data_file = child_options['base_path'] + os.sep + child_options['filename']
			self.filehandles.append(TDFileReader(data_file,child_options));
			self.filehandles[-1].read_file_info();
			
		if self.frames_per_partition < self.classes:
			logger.critical('Number of frames per partition must be greater than the number of classes, \
				Please increase the partition size');
		self.frames_per_class = self.frames_per_partition/self.classes
		
		return self.header
		
	def read_next_partition_data(self,already_read=0,pad_zeros=False):
		self.cur_frame_num = 0
		self.feat = []
		self.label = []
		none_cnt = 0
		while self.cur_frame_num < self.frames_per_partition-already_read and none_cnt < self.classes :
			if not self.filehandles[self.last_class_idx] is None:
				none_cnt = 0
				self.filehandles[self.last_class_idx].read_next_partition_data(already_read=self.frames_per_partition-self.frames_per_class)
				if self.filehandles[self.last_class_idx].cur_frame_num > 0:
					self.feat = numpy.append(self.feat,self.filehandles[self.last_class_idx].feat)
					self.label = numpy.append(self.label,self.filehandles[self.last_class_idx].label)
					self.cur_frame_num += self.filehandles[self.last_class_idx].cur_frame_num

				if self.filehandles[self.last_class_idx].is_finish():
					self.filehandles[self.last_class_idx] = None
			else:
				none_cnt = none_cnt+1;
			self.last_class_idx = (self.last_class_idx + 1)	% self.classes
			
		if self.cur_frame_num >  0:
			if pad_zeros:	#padding zeros
				self.pad_zeros(int(self.frames_per_partition)-already_read-self.cur_frame_num);
				
			logger.debug('T1 Filereader : from file %s, %d partition has %d frames' % (self.filepath,self.partition_num,self.cur_frame_num));
			self.feat = self.feat.reshape([self.cur_frame_num,self.feat_dim])
			self.feat = numpy.asarray(self.feat, dtype = theano.config.floatX)
			self.partition_num = self.partition_num + 1
			
			if not self.options['keep_flatten'] :	#reshape the vector if needed
				logger.debug('T1 Filereader : Reshape input...')
				shape = [self.cur_frame_num];
				shape.extend(self.options['input_shape']);
				self.feat = self.feat.reshape(shape);
				self.feat = dimshuffle(self.feat,self.options['dim_shuffle'])
		else:
			self.end_reading = True

##########################################NP FILEREADER##############################################################
"""
	Reads the dataset which is stored as single file in binary format
	<json-header>
	<feat_vector>	> stored as structured numpy.array
	.
	.
"""			

class NPFileReader(FileReader):
	''' Reads the data stored in as Numpy Array'''
	def __init__(self,path,options):
		self.filepath = path;
		self.options=options;
		self.batch_size = options['batch_size']
		self.filehandle = open(self.filepath,'rb')
		
	def read_file_info(self):
		jsonheader = self.filehandle.readline();
		self.header = json.loads(jsonheader);
		self.feat_dim = self.header['featdim'];
		logger.debug("NP Filereader : feats : %d"% self.feat_dim);
		# partitions specifies approximate amount data to be loaded one operation
		self.frames_per_partition= self.options['partition'] *1000*1000/ (self.feat_dim * 4)
		batch_residual = self.frames_per_partition% self.batch_size
		self.frames_per_partition= self.frames_per_partition- batch_residual
		self.dtype = numpy.dtype({'names': ['d','l'],'formats': [('>f2',self.feat_dim),'>i2']})
		return self.header
	
	def read_next_partition_data(self,already_read=0,pad_zeros=False):
		data = numpy.fromfile(self.filehandle,dtype=self.dtype,count=self.frames_per_partition);
		self.cur_frame_num = data.__len__();
		if self.cur_frame_num > 0:
			self.feat = numpy.asarray(data['d'], dtype = theano.config.floatX)
			self.label = numpy.asarray(data['l'], dtype = theano.config.floatX)
			self.partition_num = self.partition_num + 1
			if pad_zeros:
				self.pad_zeros(int(self.frame_per_partition)-self.cur_frame_num);
				for x in xrange(self.num_pad_frames):
					self.label = numpy.append(self.label,[0]*self.feat_dim) 
			
			logger.debug('NP Filereader : from file %s, %d partition has %d frames',
				self.filepath,self.partition_num,self.cur_frame_num);
			
			if not self.options['keep_flatten'] :	#reshape the vector if needed
				logger.debug('NP Filereader : Reshape input...')
				shape = [self.cur_frame_num];
				shape.extend(self.header['input_shape']);
				self.feat = self.feat.reshape(shape);
				self.feat = dimshuffle(self.feat,self.options['dim_shuffle'])
		else:
			self.end_reading = True;
		


		
