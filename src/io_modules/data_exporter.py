import json,numpy,sys,os,shutil
import theano
import theano.tensor as T

from utils.utils import dimshuffle
from io_modules.file_writer import write_dataset
from io_modules.file_reader import read_dataset
from io_modules import create_folder_structure_if_not_exists

import logging
logger = logging.getLogger(__name__)


def export_data(data_spec,export_path,out_fn,out_featdim):
	logger.info("%s data exporter will be initialized to export to %s",
				data_spec['reader_type'],export_path);
	exporter = DataExporter.get_instance(data_spec,export_path);
	exporter.dump_data(out_fn,out_featdim);

##########################################BASE CLASS##############################################################
'''
	DataExporter class focuses on exporting the features which is derived from neural network model
'''	

class DataExporter(object):
	'''Exports the data just like the input available'''
	data_spec = None
	export_path = None
	@staticmethod
	def get_instance(data_spec,export_path):
		if(data_spec['reader_type']=='T1'):
			return T1DataExporter(data_spec,export_path);
		elif(data_spec['reader_type']=='T2'):
			return T2DataExporter(data_spec,export_path);	
		elif(data_spec['reader_type']=='NP'):
			return NPDataExporter(data_spec,export_path);
		else:
			logger.critical(" '%s\'  reader_type is not defined...",data_spec['reader_type'])
	def dump_data(self,out_fn):
		pass


##########################################T1 DataExporter##############################################################
'''
	T1DataExporter focuses on exporting the features which is derived from neural network model where datset is \
	represented as T1 Dataset
'''	

class  T1DataExporter(DataExporter):
	def __init__(self,data_spec,export_path):
		self.data_spec = data_spec;
		self.export_path = export_path;
		
	def dump_data(self,out_fn,out_featdim):
		filepath = self.data_spec['base_path'] + os.sep + self.data_spec['filename']
		
		copy_path = create_folder_structure_if_not_exists(self.export_path + os.sep + self.data_spec['filename'])
		shutil.copy(filepath,copy_path);	#copies the file directly
		
		self.filehandle = open(filepath,'rb')
		line = self.filehandle.readline(); # reading file header
		header = line.split();
		num_classes = int(header[1]);
		
		for idx in xrange(num_classes):
			in_child_options = self.data_spec.copy();
			in_child_options['filename'] = self.filehandle.readline().strip()	#filename of individual classes
			in_child_options['reader_type'] = "TD"
			in_child_options['label'] = idx;
			file_reader  = read_dataset(in_child_options,pad_zeros=True)	#taking only one reader 
			out_child_options = in_child_options.copy();
			out_child_options['base_path'] = self.export_path;
			out_child_options['featdim'] = out_featdim;
			out_child_options['writer_type'] = "TD"
			file_writer =  write_dataset(out_child_options);
			batch_size=file_reader.batch_size
			
			while (not file_reader.is_finish()):
				for batch_index in xrange(file_reader.nBatches):
					s_idx = batch_index*batch_size;
					e_idx = s_idx + batch_size
					data = out_fn(file_reader.feat[s_idx:e_idx])
					label = file_reader.label[s_idx:e_idx];

					if ((batch_index == file_reader.nBatches-1) and (not file_reader.num_pad_frames == 0)) :
						data=data[:-file_reader.num_pad_frames]
						label = label[:-file_reader.num_pad_frames]

					file_writer.write_data(data,label);
				
				file_reader.read_next_partition_data(pad_zeros=True);
				
		logger.debug('T1 Dataexporter : data is exported to %s' % self.export_path);
		
		
##########################################T2 DataExporter##############################################################
'''
	T2DataExporter focuses on exporting the features which is derived from neural network model where datset is \
	represented as T2 Dataset
'''	

class  T2DataExporter(DataExporter):
	def __init__(self,data_spec,export_path):
		self.data_spec = data_spec;
		self.export_path = export_path;
		
	def dump_data(self,out_fn,out_featdim):
		filepath = self.data_spec['base_path'] + os.sep + self.data_spec['filename']
		
		copy_path = create_folder_structure_if_not_exists(self.export_path + os.sep + self.data_spec['filename'])
		shutil.copy(filepath,copy_path);	#copies the file directly
		
		self.filehandle = open(filepath,'rb')
		line = self.filehandle.readline(); # reading file header
		header = line.split();
		num_classes = int(header[1]);
		
		for idx in xrange(num_classes):
			level1_filename = self.filehandle.readline().strip();
			level1_filepath  = self.data_spec['base_path'] + os.sep + level1_filename	#filename of individual classes
			
			copy_path = create_folder_structure_if_not_exists(self.export_path + os.sep + level1_filename)
			shutil.copy(level1_filepath,copy_path);	#copies the index file directly from the source directly
			
			self.level1FileHandle = open(level1_filepath,'rb');
			level2_filepath = self.level1FileHandle.readline().strip();
			while level2_filepath.__len__()!= 0:			
				in_child_options = self.data_spec.copy();
				in_child_options['filename'] = level2_filepath	#filename of individual classes
				in_child_options['reader_type'] = "TD"
				in_child_options['label'] = idx;
				file_reader  = read_dataset(in_child_options,pad_zeros=True)	#taking only one reader 
				out_child_options = in_child_options.copy();
				out_child_options['base_path'] = self.export_path;	#updating the base_path
				out_child_options['featdim'] = out_featdim;
				out_child_options['writer_type'] = "TD"
				file_writer =  write_dataset(out_child_options);
				batch_size=file_reader.batch_size

				while not file_reader.is_finish():
					for batch_index in xrange(file_reader.nBatches):
						s_idx = batch_index * batch_size; e_idx = s_idx + batch_size
						data = out_fn(file_reader.feat[s_idx:e_idx])
						label = file_reader.label[s_idx:e_idx];

						if ((batch_index == file_reader.nBatches-1) and (not file_reader.num_pad_frames == 0)) :
							data=data[:-file_reader.num_pad_frames]
							label = label[:-file_reader.num_pad_frames]

						file_writer.write_data(data,label);
					
					file_reader.read_next_partition_data(pad_zeros=True);
			
				level2_filepath = self.level1FileHandle.readline().strip();
		logger.debug('T2 Dataexporter : data is exported to %s' % self.export_path);
		
		
##########################################NP DataExporter##############################################################
'''
	NPDataExporter focuses on exporting the features which is derived from neural network model where datset is \
	represented as NP Dataset
'''	

class  NPDataExporter(DataExporter):
	def __init__(self,data_spec,export_path):
		self.data_spec = data_spec;
		self.export_path = export_path;
		
	def dump_data(self,out_fn,out_featdim):
		file_reader  = read_dataset(self.data_spec,pad_zeros=True)	#taking only one reader 
		out_options = self.data_spec.copy();
		out_options['base_path'] = self.export_path;	#updating the base_path
		out_options['featdim'] = out_featdim;
		out_options['writer_type'] = "NP"
		file_writer =  write_dataset(out_options);
		batch_size=file_reader.batch_size

		while not file_reader.is_finish():
			for batch_index in xrange(file_reader.nBatches):
				s_idx = batch_index * batch_size; e_idx = s_idx + batch_size
				data = out_fn(file_reader.feat[s_idx:e_idx])
				label = file_reader.label[s_idx:e_idx];

				if ((batch_index == file_reader.nBatches-1) and (not file_reader.num_pad_frames == 0)) :
					data=data[:-file_reader.num_pad_frames]
					label = label[:-file_reader.num_pad_frames]

				file_writer.write_data(data,label);

			file_reader.read_next_partition_data(pad_zeros=True);
		logger.debug('NP Dataexporter : data is exported to %s' % self.export_path);
