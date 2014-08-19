import json,numpy,sys,os
from utils.utils import dimshuffle
import theano
import theano.tensor as T

import logging
logger = logging.getLogger(__name__)




def export_data(data_spec,out_fn,export_path):
	exporter = DataExporter.get_instance(data_spec,export_path);
	exporter.dump_data(out_fn,batch_size);
	


class DataExporter(object):
	data_spec = None
	export_path = None
	@staticmethod
	def get_instance(data_spec,export_path):
		if(data_spec['reader_type']=='T1'):
			return T1DataExporter(data_spec,export_path);
		else:
			return NPDataExporter(data_spec,export_path);
	def dump_data(self,out_fn):
		pass

class  T1DataExporter(DataExporter):
	def __init__(self,data_spec,export_path):
		self.data_spec = data_spec;
		self.export_path = self.export_path;
	def dump_data(self,out_fn):
		filepath = self.data_spec['path']
		self.filehandle = open(self.filepath,'rb')
		# File Header
		line = self.filehandle.readline();
		
		
		
		
	
def write_dataset(options):
	file_path = options.pop('filename');
	file_writer = FileWriter.get_instance(file_path,options);
	file_writer.write_file_info()
	return file_writer

class FileWriter(object):
	@staticmethod
	def get_instance(path,header):
		if(header['writer_type']=='T1'):
			return T1FileWriter(path,header);
		else:
			return NPFileWriter(path,header);
	def write_file_info(self):
		pass
	def write_data(self,vector_array,labels):
		pass


class NPFileWriter(FileWriter):
	def __init__(self,filename,header):
		self.header = header
		self.filepath = header['path']+filename
		self.filehandle = open(self.filepath,'ab')

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

class TDFileWriter(FileWriter):
	def __init__(self,filename,header):
		self.header = header
		self.filepath = header['path']+filename
		self.filehandle = open(self.filepath,'a+')

	def write_file_info(self):
		self.filehandle.write(header['featdim']+os.linesep)

	def write_data(self,vector,label):
		for element in vector:
			self.filehandle.write('%d ' % element)
		self.filehandle.write(os.linesep);
		self.filehandle.flush();		

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
			path = self.prefix + label+".data"
			filehandle.write(path + os.linesep);
			self.childhandles.append(TDFileWriter(path,self.header));
			self.childhandles[-1].write_file_info();
		self.filehandle.flush()
		
	def write_data(self,vectors,labels):
		featdim = self.header[0];
		for vector,label in zip(vectors,labels):
			flatten_vector = vector.flatten();
			if featdim==len(flatten_vector):
				logger.critical('Feature dimension mentioned in header and vector length is mismatching');
				self.childhandles[label].writedata(vector,label); 

