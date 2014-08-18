import json,numpy,sys
import theano
import theano.tensor as T
from utils.utils import dimshuffle

def read_dataset(options):
	file_path = options.pop('path');
	file_reader = FileReader.get_instance(file_path,options)
	file_header = file_reader.read_file_info()

	shared_xy = file_reader.create_shared()
	shared_x, shared_y = shared_xy
	shared_y = T.cast(shared_y, 'int32')

	return file_reader,shared_xy, shared_x, shared_y


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
	frames_remaining = 0
	end_reading = False
	
	@staticmethod
	def get_instance(filepath,options):
		file_reader = None;
		if options['reader_type']=='NP':
			file_reader = NPFileReader(filepath,options);
		elif options['reader_type']=='T':
			file_reader = TFileReader(filepath,options);
		return file_reader
	'''Reads the file header information'''	
	def read_file_info(self):
		pass
	'''Reads the data from the next partition'''	
	def read_next_partition_data(self,already_read=0):
		pass
	'''Makes the current partition shared across GPU's'''	
	def make_partition_shared(self, shared_xy):
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
			
	'''Create first partition shared across GPU's'''	
	def create_shared(self):
		if self.feat is None:
			self.read_next_partition_data()
		shared_x = theano.shared(self.feat, name = 'x', borrow = True)
		shared_y = theano.shared(self.label, name = 'y', borrow = True)
		return shared_x, shared_y
	'''Reopens the file to read from start'''	
	def reopen_file(self):
		self.filehandle = open(self.file_path)
		initialize_read();
	'''Checks if the file reading reached end of file '''
	def is_finish(self):
		return self.end_reading
	''' Initialize the file_reader options''' 	
	def initialize_read(self):		
		self.filehandle.seek(0,0);
		self.end_reading = False
		self.feat=None
		self.labels=None
		self.partition_num = 0
		self.frame_per_partition = 0	
		self.read_file_info()
		self.read_next_partition_data()		

class TDFileReader(FileReader):
	''' Reads the data stored in as Simple Text File'''
	def __init__(self,path,options):
		self.filepath = path;
		self.options=options;
		self.lbl = options.pop('label');
		self.filehandle = open(self.filepath,'rb')		
		
	def read_file_info(self):
		header = self.filehandle.readline();
		self.header = header.split()
		self.feat_dim = int(self.header[0]);
		self.frames_remaining = long(self.header[1])
		# partitions specifies approximate amount data to be loaded one operation
		self.frame_per_partition = self.options['partition'] *1000*1000/ (self.feat_dim * 4)
		batch_residual = self.frame_per_partition % self.options['batch_size']
		self.frame_per_partition = self.frame_per_partition - batch_residual
		return self.header
		
	def read_next_partition_data(self,already_read=0):
		self.feat=[]
		self.cur_frame_num = 0
		while self.frames_remaining > 0 and self.cur_frame_num < self.frame_per_partition-already_read:
			values = self.filehandle.readline().split()
			fvalues = [float(value) for value in values];
			self.feat = numpy.append(self.feat,fvalues)
			self.frames_remaining -=1
			self.cur_frame_num += 1
		
		if self.cur_frame_num >  0:
			self.feat = self.feat.reshape([self.cur_frame_num,self.feat_dim])
			self.label = numpy.asarray([self.lbl]*self.cur_frame_num , dtype = theano.config.floatX)
			self.partition_num = self.partition_num + 1

		if self.frames_remaining == 0:
			self.end_reading = True;

class TFileReader(FileReader):
	''' Reads the data stored in as Simple Text File'''
	def __init__(self,path,options):
		self.filepath = path;
		self.options=options;
		self.filehandle = open(self.filepath,'rb')
		self.childfilehandle=None;
		
	def read_file_info(self):
		header = self.filehandle.readline();
		self.header = header.split()
		self.feat_dim = int(self.header[0]);
		self.files_remaining = long(self.header[1])
		# partitions specifies approximate amount data to be loaded one operation
		self.frame_per_partition = self.options['partition'] *1000*1000/ (self.feat_dim * 4)
		batch_residual = self.frame_per_partition % self.options['batch_size']
		self.frame_per_partition = self.frame_per_partition - batch_residual
		return self.header
		
	def read_next_partition_data(self,already_read=0):
		self.cur_frame_num = 0
		self.feat=[]
		self.label = []
		if self.files_remaining == 0:
			self.end_reading = True;
	
		while self.files_remaining>0 and self.cur_frame_num < self.frame_per_partition-already_read:
			if self.childfilehandle is None:
				data_file = self.filehandle.readline().split();
				self.options['label']=int(data_file[1])
				self.childfilehandle = TDFileReader(data_file[0],self.	options)
				self.childfilehandle.read_file_info()
			
			self.childfilehandle.read_next_partition_data(already_read=self.cur_frame_num)
			
			if self.childfilehandle.cur_frame_num > 0:
				self.feat = numpy.append(self.feat,self.childfilehandle.feat)
				self.label = numpy.append(self.label,self.childfilehandle.label)
				self.cur_frame_num += self.childfilehandle.cur_frame_num

			if self.childfilehandle.is_finish():
				self.childfilehandle = None
				self.files_remaining -=1
			
		if self.cur_frame_num >  0:
			self.feat = self.feat.reshape([self.cur_frame_num,self.feat_dim])
			self.feat = numpy.asarray(self.feat, dtype = theano.config.floatX)
			
			self.partition_num = self.partition_num + 1
			if not self.options['keep_flatten'] :	#reshape the vector if needed
				shape = [self.cur_frame_num];
				shape.extend(self.options['input_shape']);
				self.feat = self.feat.reshape(shape);
				self.feat = dimshuffle(self.feat,self.options['dim_shuffle'])
#			else :
#				self.feat = self.feat.flatten(); 

		
class NPFileReader(FileReader):
	''' Reads the data stored in as Numpy Array'''
	def __init__(self,path,options):
		self.filepath = path;
		self.options=options;
		self.filehandle = open(self.filepath,'rb')
		
	def read_file_info(self):
		jsonheader = self.filehandle.readline();
		self.header = json.loads(jsonheader);
		self.feat_dim = self.header['featdim'];
		# partitions specifies approximate amount data to be loaded one operation
		self.frame_per_partition = self.options['partition'] *1000*1000/ (self.feat_dim * 4)
		batch_residual = self.frame_per_partition % self.options['batch_size']
		self.frame_per_partition = self.frame_per_partition - batch_residual
		self.dtype = numpy.dtype({'names': ['d','l'],'formats': [('>f2',self.feat_dim),'>i2']})
		return self.header

	def read_next_partition_data(self,already_read=0):
		data = numpy.fromfile(self.filehandle,dtype=self.dtype,count=self.frame_per_partition);
		self.cur_frame_num = data.__len__();
		if self.cur_frame_num > 0:
			self.feat = numpy.asarray(data['d'], dtype = theano.config.floatX)
			self.label = numpy.asarray(data['l'], dtype = theano.config.floatX)
			self.partition_num = self.partition_num + 1
			if not self.options['keep_flatten'] :	#reshape the vector if needed
				shape = [self.cur_frame_num];
				shape.extend(self.header['input_shape']);
				self.feat = self.feat.reshape(shape);
				self.feat = dimshuffle(self.feat,self.options['dim_shuffle'])
#			else :
#				self.feat = self.feat.flatten(); 
		else:
			self.end_reading = True;


		
