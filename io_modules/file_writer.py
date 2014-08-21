import json,numpy,sys,os
from utils.utils import dimshuffle
import theano
import theano.tensor as T

import logging
logger = logging.getLogger(__name__)
	
def write_dataset(options):
	file_path = options['base_path'] + os.sep + options['filename'];
	file_writer = FileWriter.get_instance(file_path,options);
	file_writer.write_file_info()
	return file_writer

##########################################BASE CLASS##############################################################

class FileWriter(object):
	header = None
	feat_dim = None
	filepath = None
	filehandle = None
	
	@staticmethod
	def get_instance(path,header):
		if(header['writer_type']=='T1'):
			return T1FileWriter(path,header);
		elif header['writer_type']=='TD':
			return TDFileWriter(path,header);
		elif header['writer_type']=='NP':
			return NPFileWriter(path,header);
		else:
			logger.critical('\'%s\'  writer_type is not defined...')
			
	def __init__(self,filepath,header):
		self.header = header
		self.filepath = filepath
		self.feat_dim = header['featdim'];
		self.filehandle = open(self.filepath,'ab')
		
	def write_file_info(self):
		pass
	def write_data(self,vector_array,labels):
		pass
		
##########################################NP Filewriter##############################################################
"""
	writes the dataset which is stored as single file in binary format
	<json-header>
	<feat_vector>	> stored as structured numpy.array
	.
	.
"""

class NPFileWriter(FileWriter):
	def __init__(self,filepath,header):
			super(NPFileWriter,self).__init__(filepath,header)

	def write_file_info(self):
		self.filehandle.write(json.dumps(self.header)+'\n')

	def write_data(self,vector_array,labels):
		dt={'names': ['d','l'],'formats': [('>f2',self.feat_dim),'>i2']}
		data = numpy.zeros(1,dtype= numpy.dtype(dt))
		for vector,label in zip(vector_array,labels):
			flatten_vector = vector.flatten();
			if self.feat_dim==len(flatten_vector):
				data['d']=flatten_vector; data['l']=label;
				data.tofile(self.filehandle); 
		self.filehandle.flush();

##########################################TD FILEREADER##############################################################
"""
	Reads the simple text file following is the structure
	<feat_dim> <num_feat_vectors>(optional)
	<feat_vector>
	<feat_vector>
	.
	.
"""
class TDFileWriter(FileWriter):
	def __init__(self,filepath,header):
		super(TDFileWriter,self).__init__(filepath,header)

	def write_file_info(self):
		self.filehandle.write(str(self.feat_dim)+os.linesep)

	def write_data(self,vector_array,labels):
		for vector,label in zip(vector_array,labels):
			flatten_vector = vector.flatten();
			if self.feat_dim==len(flatten_vector):
				logger.critical('Feature dimension mentioned in header and vector length is mismatching');
			for element in vector:
				self.filehandle.write('%f ' % element)
			self.filehandle.write(os.linesep);
		self.filehandle.flush();		


"""
class T1FileWriter(FileWriter):
	def __init__(self,path,header):
		self.header = header
		self.filepath = path
		self.filehandle = open(self.filepath,'a+')
		self.childhandles = []

	def write_file_info(self):
		self.filehandle.write(('%d %d'+os.linesep) % (self.header['featdim'],self.header['classes']))
		self.path = self.header['path'];
		
		for idx in xrange(0,self.classes):
			child_header = self.header.copy();
			child_header['filename'] = "%d.data"%label
			path = child_header['base_path'] + os.sep + child_header['filename'] 
			filehandle.write(path + os.linesep);
			self.childhandles.append(TDFileWriter(path,child_header));
			self.childhandles[-1].write_file_info();
		self.filehandle.flush()
		
	def write_data(self,vectors,labels):
		featdim = self.header[0];
		for vector,label in zip(vectors,labels):
			flatten_vector = vector.flatten();
			if featdim==len(flatten_vector):
				logger.critical('Feature dimension mentioned in header and vector length is mismatching');
				self.childhandles[label].writedata(vector,label); 

"""
