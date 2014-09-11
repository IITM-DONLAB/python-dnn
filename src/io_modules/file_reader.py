import json,numpy,sys,os
import theano
import theano.tensor as T
from utils.utils import dimshuffle

import logging
logger = logging.getLogger(__name__)

def read_dataset(options,pad_zeros=False,makeShared=True):
	filepath =   options['base_path'] + os.sep + options['filename'];
	logger.info("%s dataset will be initialized to reader to %s",
				options['reader_type'],filepath);

	file_reader = FileReader.get_instance(filepath,options)
	#file_header = file_reader.read_file_info()
	file_reader.read_next_partition_data(
		pad_zeros=pad_zeros,
		makeShared=makeShared);
	return file_reader

##########################################BASE CLASS##############################################################

class FileReader(object):
	'''Gets instance of file based on the reader type'''

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
			logger.critical("'%s' reader_type is not defined...",options['reader_type'])
		return file_reader

	def __init__(self):
		self.filepath = None;
		self.options=None;
		# store features and labels for each data partition
		self.feat = None
		self.label = None
		# markers while reading data
		self.end_reading = False
		self.feat=None
		self.labels=None
		self.feat_dim = 0;
		self.partition = 0;
		self.partition_num = 0
		self.frames_per_partition= 0
		self.num_pad_frames = 0;
		self.cur_frame_num = 0;
		self.nBatches = 0
		self.made_shared = False;


	def read_file_info(self):
                '''Reads the file header information'''
		pass

	def skipHeader(self):
                '''skip header in data'''
		pass

	def read_next_partition_data(self,already_read=0,pad_zeros=False,makeShared=True):
                '''Reads the data from the next partition'''
		pass

	def make_partition_shared(self):
                '''Makes the current partition shared across GPU's'''
		if not self.made_shared:
			self.create_shared();

		(shared_x, shared_y) = self.shared_xy;
		logger.debug("Partition is now made shared for GPU processing");
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

	def setPartitionFrames(self):
                '''Set approximate amount data to be loaded one partition'''
		# partitions specifies approximate amount data to be loaded one operation
		if self.partition == 0:
			# if partition is zero,partition has NO LIMIT!!!!
			self.frames_per_partition = float('inf');
		else:
			self.frames_per_partition= (self.partition*1000*1000)/ (self.feat_dim * 4)
			batch_residual = self.frames_per_partition% self.batch_size
			self.frames_per_partition = self.frames_per_partition - batch_residual

	def pad_zeros(self):
                """
                Padd Zeros(if required) to remaining batch size
                """
                ## framesRemains = batch_size*(cur_frame_num/batch_size +1) - cur_frame_num
                ## framesRemains = (batch_size - (cur_frame_num % batch_size))%batch_size
                ## framesRemains = (batch_size%batch_size - (cur_frame_num % batch_size)%batch_size)%batch_size
                ## framesRemains = ( 0 - (cur_frame_num % batch_size))%batch_size
                ## framesRemains = (-1*cur_frame_num) % batch_size)
                
                framesRemains = (-1*self.cur_frame_num) % self.batch_size;
		
                if framesRemains > 0:
			logger.debug("Padded %d frames for one partition",framesRemains);
			self.num_pad_frames = framesRemains;
			padding = numpy.zeros((framesRemains,self.feat_dim),dtype = theano.config.floatX)
                        try:
                                #try append verticaly
                                self.feat = numpy.append(self.feat,padding,axis=0)
                        except ValueError:
                                self.feat = numpy.append(self.feat,padding)
			self.label = numpy.append(self.label,[0]*framesRemains);
			self.cur_frame_num+=framesRemains


	def create_shared(self):
                '''Create first partition shared across GPU's'''
		shared_x = theano.shared(self.feat, name = 'x', borrow = True)
		shared_y = theano.shared(self.label, name = 'y', borrow = True)
		self.shared_xy = (shared_x, shared_y)
		self.shared_y = T.cast(shared_y, 'int32')
		self.shared_x = shared_x;

		self.made_shared = True;

	def is_finish(self):
                '''Checks if the file reading reached end of file '''
		return self.end_reading

	def initialize_read(self,makeShared=True):
                ''' Initialize the file_reader options'''
		logger.debug("File reader is initialzed again for file reading");
		self.end_reading = False
		if self.partition_num == 1:
			logger.debug("File reader:Only one partition keep same.");
		else:
			self.num_pad_frames = 0
			#self.frames_per_partition = 0
			self.partition_num = 0
			self.label = None
			self.feat = None
			self.skipHeader()
			self.read_next_partition_data(makeShared=makeShared)

        def __len__(self):
                return self.cur_frame_num;


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
		super(NPFileReader,self).__init__();
		self.filepath = path;
		self.options=options;
		self.batch_size = options['batch_size']
		self.partition = options['partition']
		self.filehandle = open(self.filepath,'rb')
		self.read_file_info();

	def read_file_info(self):
		jsonheader = self.filehandle.readline();
		self.header = json.loads(jsonheader);
		self.header_size = len(jsonheader);
		self.feat_dim = self.header['featdim'];
		logger.debug("NP Filereader : feats : %d"% self.feat_dim);
		# partitions specifies approximate amount data to be loaded one operation
		self.setPartitionFrames()
		self.dtype = numpy.dtype({'names': ['d','l'],'formats': [('>f2',self.feat_dim),'>i2']})

	def skipHeader(self):
		self.filehandle.seek(self.header_size,0)
		

	def read_next_partition_data(self,already_read=0,pad_zeros=False,makeShared=True):
		data = numpy.fromfile(self.filehandle,dtype=self.dtype,count=self.frames_per_partition);
		cur_frame_num = data.__len__();

		if cur_frame_num > 0:
			self.cur_frame_num = cur_frame_num
			self.feat = numpy.asarray(data['d'], dtype = theano.config.floatX)
			self.label = numpy.asarray(data['l'], dtype = theano.config.floatX)
			self.partition_num = self.partition_num + 1
			if pad_zeros:
				self.pad_zeros();

			if not self.options['keep_flatten'] :	#reshape the vector if needed
				logger.debug('NP Filereader : Reshape input...')
				shape = [self.cur_frame_num];
				shape.extend(self.header['input_shape']);
				self.feat = self.feat.reshape(shape);
				self.feat = dimshuffle(self.feat,self.options['dim_shuffle'])
			if makeShared:
				self.make_partition_shared();
			self.nBatches = self.cur_frame_num/self.batch_size;
                        logger.debug('T1 Filereader : from file %s, %d partition has %d frames [%d Batches]',
                                     self.filepath,self.partition_num-1,self.cur_frame_num,self.nBatches);
                        
		else:
			self.end_reading = True;
                        logger.debug('TD Filereader : NO more frames to read from %s',self.filepath)
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
		super(TDFileReader,self).__init__();
		self.filepath = path;
		self.options = options;
		self.batch_size = options['batch_size']
		self.partition = options['partition']
		self.lbl = options['label'];
		self.filehandle = open(self.filepath,'rb')
		self.read_file_info();

	def read_file_info(self):
		header = self.filehandle.readline();
		
		self.header = header.split()
		self.header_size = len(header);

		self.feat_dim = int(self.header[0]);
		self.header = {}
		self.header['featdim'] = self.feat_dim
		logger.debug('TD Filereader : feats : %d, label : %d' % (self.feat_dim,self.lbl))
		#self.frames_remaining = long(self.header[1])
		# partitions specifies approximate amount data to be loaded one operation
		self.setPartitionFrames()

	def skipHeader(self):
		self.filehandle.seek(self.header_size,0)

	def read_next_partition_data(self,already_read=0,pad_zeros=False,makeShared=True):

		if self.end_reading:
			raise EOFError('Reader has reached EOF');
		#self.feat=[]
		fvalues = []
		cur_frame_num = 0
		while  cur_frame_num < self.frames_per_partition-already_read:
			values = self.filehandle.readline().split()
			if values.__len__()==0: #No more values available in the data file
				break;
			fvalues=numpy.append(fvalues,[float(value) for value in values]);
			cur_frame_num += 1

		if cur_frame_num  > 0:
			
			self.cur_frame_num = cur_frame_num;
			self.feat = fvalues;

			if pad_zeros:
				self.pad_zeros();
			
			self.feat = self.feat.reshape([self.cur_frame_num,self.feat_dim])
			# convert float64 to floatX
			self.feat = numpy.asarray(self.feat, dtype = theano.config.floatX)

			self.label = numpy.asarray([self.lbl]*self.cur_frame_num , dtype = theano.config.floatX)
			self.partition_num = self.partition_num + 1
			logger.debug('TD Filereader : %d frames read from %s',self.cur_frame_num,self.filepath)
			if not self.options['keep_flatten'] :	#reshape the vector if needed
				logger.debug('TD Filereader : Reshape input...')
				shape = [self.cur_frame_num];
				shape.extend(self.options['input_shape']);
				self.feat = self.feat.reshape(shape);
				self.feat = dimshuffle(self.feat,self.options['dim_shuffle'])

			if makeShared:
				self.make_partition_shared();
			self.nBatches = self.cur_frame_num/self.batch_size;
		else:
                        logger.debug('TD Filereader : NO more frames to read from %s',self.filepath)
			self.end_reading = True;


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
		super(T1FileReader,self).__init__();
		self.filepath = path;
		self.options=options;
		self.batch_size = options['batch_size']
		self.partition = options['partition']
		self.read_file_info();

	def read_file_info(self):
		# initializes multiple class handles
		self.filehandles = [];
		self.last_class_idx = 0;

		filehandle = open(self.filepath,'rb')
		header = filehandle.readline().split();
		self.feat_dim = int(header[0]);
		self.classes = long(header[1])

		self.header = {};
		self.header['featdim'] = self.feat_dim
		batch_size = self.batch_size

		logger.debug('T1 Filereader : feat : %d' % self.feat_dim)

		#load filehandle for all classes
		for i in xrange(0,self.classes):
			child_options = self.options.copy();
			child_options['filename'] = filehandle.readline().strip();
			child_options['label']= i;
			child_options['keep_flatten'] = True
			data_file = child_options['base_path'] + os.sep + child_options['filename']
			self.filehandles.append(TDFileReader(data_file,child_options));

		filehandle.close();

		# partitions specifies approximate amount data to be loaded one operation
		self.setPartitionFrames()

		if self.frames_per_partition < self.classes:
			logger.critical("Number of frames per partition must be greater than the number of classes,"
				"Please increase the partition size");
			exit(0);
		self.frames_per_class = self.frames_per_partition/self.classes

		return self.header

	def read_next_partition_data(self,already_read=0,pad_zeros=False,makeShared=True):
		if self.end_reading:
			raise EOFError('Reader has reached EOF');
		
		cur_frame_num = 0
		feat = []
		label = []
		while cur_frame_num < self.frames_per_partition-already_read :

			if not self.filehandles[self.last_class_idx].end_reading:
				#if TD is  not finshed.
				# Read Next part.
				logger.debug("T1: loading %s ",self.filehandles[self.last_class_idx].filepath) 
				self.filehandles[self.last_class_idx].read_next_partition_data(
					already_read=self.frames_per_partition-self.frames_per_class,
					makeShared=False)
				if self.filehandles[self.last_class_idx].end_reading:
                                        pass
                                else:
                                        feat = numpy.append(feat,self.filehandles[self.last_class_idx].feat)
                                        label = numpy.append(label,self.filehandles[self.last_class_idx].label)
                                        cur_frame_num += self.filehandles[self.last_class_idx].cur_frame_num

			self.last_class_idx = (self.last_class_idx + 1)	% self.classes
                        
                        if all([x.end_reading for x in self.filehandles]):
                                break;

		if cur_frame_num >  0:
			self.feat = feat;
			self.label = label;
			self.cur_frame_num = cur_frame_num;

                        if pad_zeros:
				self.pad_zeros();

			self.partition_num = self.partition_num + 1

			self.feat = self.feat.reshape([self.cur_frame_num,self.feat_dim])
			self.feat = numpy.asarray(self.feat, dtype = theano.config.floatX)

			if not self.options['keep_flatten'] :	#reshape the vector if needed
				logger.debug('T1 Filereader : Reshape input...')
				shape = [self.cur_frame_num];
				shape.extend(self.options['input_shape']);
				self.feat = self.feat.reshape(shape);
				self.feat = dimshuffle(self.feat,self.options['dim_shuffle'])

			if makeShared:
				self.make_partition_shared();
			self.nBatches = self.cur_frame_num/self.batch_size;
                        logger.debug('T1 Filereader : from file %s, %d partition has %d frames [%d Batches]',
                                     self.filepath,self.partition_num-1,self.cur_frame_num,self.nBatches);
		else:
			self.end_reading = True
                        logger.debug('TD Filereader : NO more frames to read from %s',self.filepath)

	def initialize_read(self,makeShared=True):
		logger.debug("File reader is initialzed again for file reading");
		self.end_reading = False
		if self.partition_num == 1:
			logger.debug("File reader:Only one partition keep same.");
		else:
			self.num_pad_frames = 0
			#self.frames_per_partition = 0
			self.partition_num = 0
			self.label = None
			self.feat = None
			for i in xrange(self.classes):
				self.filehandles[i].initialize_read();
			#self.skipHeader()
			self.read_next_partition_data(makeShared=makeShared)




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
	"""
	Reads the data stored in as Simple Text File With Two level header structure
	NOT TESSTED;
	ALSO SLOW
	"""

	def __init__(self,path,options):
		super(T2FileReader,self).__init__();
		self.filepath = path;
		self.options=options;
		self.batch_size = options['batch_size']
		self.partition = options['partition']
		self.read_file_info()
		

	def read_file_info(self):
		# initializes multiple class handles
		self.filenames = {};
		self.childhandles=[];
		self.last_class_idx = 0;

		filehandle = open(self.filepath,'rb')
		header = filehandle.readline().split();
		self.feat_dim = int(header[0]);
		self.classes = long(header[1])

		self.header = {}
		self.header['featdim'] = self.feat_dim
		self.header['classes'] = self.classes

		logger.debug('T2 Filereader : feat : %d' % self.feat_dim)

		#load filehandle for all classes
		base_path = self.options['base_path']
		for i in xrange(0,self.classes):
			data_file = base_path + os.sep + filehandle.readline().strip();
			filenames = [line.strip() for line in open(data_file,'rb')]
			filenames = [(base_path + os.sep + name) for name in filter(None, filenames)]
			filenames.append(None);
			self.filenames[i] = filenames;

		filehandle.close();
		self.fileIndex = [0]*(self.classes);

		# partitions specifies approximate amount data to be loaded one operation
		batch_size = self.batch_size
		self.setPartitionFrames()


		if self.frames_per_partition < self.classes:
			logger.critical('Number of frames per partition must be greater than the number of classes,'
				'Please increase the partition size');
			exit(0);
		self.frames_per_class = self.frames_per_partition/self.classes


	def read_next_partition_data(self,already_read=0,pad_zeros=False,makeShared=True):
		cur_frame_num = 0
		feat = numpy.array([],dtype=theano.config.floatX)
		label = []
		none_cnt = 0

		while cur_frame_num < self.frames_per_partition and none_cnt < self.classes :
			if self.childhandles[self.last_class_idx] is None:	
				#if the child handle is not initialized
				data_file = self.filenames[self.last_class_idx][fileIndex[self.last_class_idx]];
				##Get Next Filename in last Class.
				if data_file != None:
					#if Next Filename == NULL;

					fileIndex[self.last_class_idx] = fileIndex[self.last_class_idx] + 1;
					child_options = self.options.copy()
					child_options['filename']=data_file
					child_options['label']= self.last_class_idx
					child_options['keep_flatten'] = True
					data_file = child_options['filename']
					self.childhandles[self.last_class_idx] = TDFileReader(data_file,child_options)

			if not self.childhandles[self.last_class_idx] is None:
				none_cnt = 0
				self.childhandles[self.last_class_idx].read_next_partition_data(
					already_read=self.frames_per_partition-self.frames_per_class,
					makeShared=False)
				
				if not self.childhandles[self.last_class_idx].end_reading:
					feat = numpy.append(feat,self.childhandles[self.last_class_idx].feat)
					label = numpy.append(label,self.childhandles[self.last_class_idx].label)
					cur_frame_num += self.childhandles[self.last_class_idx].cur_frame_num

				if self.childhandles[self.last_class_idx].is_finish():
					self.childhandles[self.last_class_idx] = None
			else:
				none_cnt+=1

			self.last_class_idx = (self.last_class_idx + 1)	% self.classes

		if cur_frame_num >  0:
			self.feat = feat;
			self.label = label;
			self.cur_frame_num = cur_frame_num;

			if pad_zeros:	#padding zeros
				self.pad_zeros();

			logger.debug('T2 Filereader : from file %s, %d partition has %d frames',
				self.filepath,self.partition_num,self.cur_frame_num);
			self.feat = self.feat.reshape([self.cur_frame_num,self.feat_dim])
			self.feat = numpy.asarray(self.feat, dtype = theano.config.floatX)
			self.partition_num = self.partition_num + 1

			if not self.options['keep_flatten'] :	#reshape the vector if needed
				logger.debug('T2 Filereader : Reshape input...')
				shape = [self.cur_frame_num];
				shape.extend(self.options['input_shape']);
				self.feat = self.feat.reshape(shape);
				self.feat = dimshuffle(self.feat,self.options['dim_shuffle'])
			
			if makeShared:
				self.make_partition_shared();
			self.nBatches = self.cur_frame_num/self.batch_size;
		else:
			self.end_reading = True
