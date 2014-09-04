import matplotlib
import numpy as np
import logging
from io_modules import create_folder_structure_if_not_exists
logger = logging.getLogger(__name__)

def plot(layer_output,path,layer_idx,batch_idx,img_plot_remaining,max_subplots=100):
	#print layer_output;
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	num_examples = layer_output.__len__();
	for eg_idx in xrange(num_examples):
		if img_plot_remaining == 0:
			break;
		save_path = path%(layer_idx,batch_idx,eg_idx)
		logger.debug('Plotting the feature map %d in %s'%(eg_idx,save_path));		
		eg_output = layer_output[eg_idx];
		
		num_plots = min(max_subplots,eg_output.__len__());
		cols = int(np.ceil(num_plots/10.0));
		fig,plots = plt.subplots(10,cols);
		plots = plots.flatten();
		for idx in xrange(num_plots):
			plots[idx].imshow(eg_output[idx],interpolation='nearest');
			plots[idx].axis('off');
		
		create_folder_structure_if_not_exists(save_path)
		# Save the full figure...
		fig.savefig(save_path,bbox_inches='tight')
		plt.close()
		img_plot_remaining-=1;
		
	return img_plot_remaining;
