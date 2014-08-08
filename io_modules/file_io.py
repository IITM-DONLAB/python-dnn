import json,numpy,sys
from utils.utils import dimshuffle
import theano
import theano.tensor as T

def write_dataset(options):
	file_path = options.pop('path');
	file_writer = FileWriter(file_path,options);
	file_writer.write_file_info()
	return file_writer


class FileWriter(object):
	def __init__(self,path,header):
		self.header = header
		self.filepath = path
		self.filehandle = open(self.filepath,'wb')

	def write_file_info(self):
    		self.filehandle.write(json.dumps(self.header)+'\n')

	def write_data(self,vector_array,labels):
		featdim= self.header['featdim'];
		dt={'names': ['d','l'],'formats': [('>f2',featdim),'>i2']}
		data = numpy.zeros(1,dtype= numpy.dtype(dt))
    		for vector,label in zip(vector_array,labels):
			flatten_vector = vector.flatten();
			if featdim==len(flatten_vector):
				data['d']=flatten_vector; data['l']=label;
				data.tofile(self.filehandle); 

		self.filehandle.flush();		


def read_dataset(options):
	file_path = options.pop('path');
	file_reader = FileReader(file_path,options)
	file_header = file_reader.read_file_info()

	shared_xy = file_reader.create_shared()
	shared_x, shared_y = shared_xy
	shared_y = T.cast(shared_y, 'int32')

	return file_reader,shared_xy, shared_x, shared_y

class FileReader(object):
	def __init__(self,path,options):
		self.filepath = path;
		self.options=options;
		self.filehandle = open(self.filepath,'rb')
		
		# store number of frames, features and labels for each data partition
	        self.feat = None
	        self.label = None
		self.end_reading = False
		
		# markers while reading data
		self.partition_num = 0
		self.frame_per_partition = 0

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

	def read_next_partition_data(self):
		data = numpy.fromfile(self.filehandle,dtype=self.dtype,count=self.frame_per_partition);
		self.cur_frame_num = data.__len__();
		if self.cur_frame_num > 0:
			self.feat = numpy.asarray(data['d'], dtype = theano.config.floatX)
			self.label = numpy.asarray(data['l'], dtype = theano.config.floatX)
			self.partition_num = self.partition_num + 1
			shape = [self.cur_frame_num];
			shape.extend(self.header['input_shape']);
			self.feat = self.feat.reshape(shape);
			self.feat = dimshuffle(self.feat,self.options['dim_shuffle'])
		else:
			self.feat = None
			self.end_reading = True;	

	     
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
            	print 'shape of the data.. ',self.feat.shape
		shared_x.set_value(self.feat, borrow=True)
		shared_y.set_value(self.label, borrow=True)
	
	def create_shared(self):
		if self.feat is None:
			self.read_next_partition_data()

		shared_x = theano.shared(self.feat, name = 'x', borrow = True)
		shared_y = theano.shared(self.label, name = 'y', borrow = True)
		return shared_x, shared_y

	# reopen file with the same filename
	def reopen_file(self):
		self.filehandle = open(self.file_path)
		initialize_read();

	def is_finish(self):
		return self.end_reading

	def initialize_read(self):		
		self.end_reading = False
		self.partition_num = 0
		self.feat=None
		self.filehandle.seek(0,0);
		self.read_file_info()
		self.read_next_partition_data()

