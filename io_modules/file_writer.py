import json,numpy,sys
from utils.utils import dimshuffle
import theano
import theano.tensor as T

def write_dataset(options):
	file_path = options.pop('path');
	file_writer = FileWriter.get_instance(file_path,options);
	file_writer.write_file_info()
	return file_writer

class FileWriter(object):
	def get_instance(path,header):
		if(header['file_type']=='T'):
			return TFileWriter(path,header);
		else:
			return NPFileWriter(path,header);
	def write_file_info(self):
		pass
	def write_data(self,vector_array,labels):
		pass


class NPFileWriter(FileWriter):
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

class TDFileWriter(FileWriter):
	def __init__(self,path,header):
		self.header = header
		self.filepath = path
		self.filehandle = open(self.filepath,'ab')

	def write_file_info(self):
			self.filehandle.write(json.dumps(self.header)+'\n')

	def write_data(self,vector,labels):
		featdim= self.header['featdim'];
		data = numpy.zeros(1,dtype= numpy.dtype(dt))
		for vector,label in zip(vector_array,labels):
			flatten_vector = vector.flatten();
			if featdim==len(flatten_vector):
				data['d']=flatten_vector; data['l']=label;
				data.tofile(self.filehandle); 
		self.filehandle.flush();		




class TFileWriter(FileWriter):
	def __init__(self,path,header):
		self.header = header
		self.filepath = path
		self.filehandle = open(self.filepath,'wb')

	def write_file_info(self):
			self.filehandle.write(json.dumps(self.header)+'\n')

	def write_data(self,vector_array,labels):
		
		featdim= self.header['featdim'];
		data = numpy.zeros(1,dtype= numpy.dtype(dt))
		for vector,label in zip(vector_array,labels):
			
		self.filehandle.flush();		


