import theano.tensor as T
from theano.tensor.signal.downsample import DownsampleFactorMax
import numpy as np
def max_pool_3d(input, ds, ignore_border=False):
	"""
		Takes as input a N-D tensor, where N >= 3. It downscales the input video by
		the specified factor, by keeping only the maximum value of non-overlapping
		patches of size (ds[0],ds[1],ds[2]) (time, height, width)  
		
		:type input: N-D theano tensor of input images.
		:param input: input images. Max pooling will be done over the 3 last dimensions.
		:type ds: tuple of length 3
		:param ds: factor by which to downscale. (2,2,2) will halve the video in each dimension.
		:param ignore_border: boolean value. Example when True, (5,5,5) input with ds=(2,2,2) will generate a
		(2,2,2) output. (3,3,3) otherwise.
	"""
	if input.ndim < 3:
		raise NotImplementedError('max_pool_3d requires a dimension >= 3')
		
	vid_dim = input.ndim
	#Maxpool frame
	frame_shape = input.shape[-2:]

	# count the number of "leading" dimensions, store as dmatrix
	batch_size = T.prod(input.shape[:-2])
	batch_size = T.shape_padright(batch_size,1)
	new_shape = T.cast(T.join(0, batch_size,T.as_tensor([1,]),frame_shape), 'int32')
	
	input_4D = T.reshape(input, new_shape, ndim=4)
	# downsample mini-batch of videos in rows and cols
	op = DownsampleFactorMax((ds[1],ds[2]), ignore_border)
	output = op(input_4D)
	# restore to original shape
	outshape = T.join(0, input.shape[:-2], output.shape[-2:])
	out = T.reshape(output, outshape, ndim=input.ndim)
	
	#Maxpool time 
	# output (time, rows, cols), reshape so that time is in the back
	shufl = (list(range(vid_dim-4)) + list(range(vid_dim-3,vid_dim))+[vid_dim-4])
	input_time = out.dimshuffle(shufl)
	# reset dimensions
	vid_shape = input_time.shape[-2:]
	# count the number of "leading" dimensions, store as dmatrix
	batch_size = T.prod(input_time.shape[:-2])
	batch_size = T.shape_padright(batch_size,1)
	# store as 4D tensor with shape: (batch_size,1,width,time)
	new_shape = T.cast(T.join(0, batch_size,T.as_tensor([1,]),vid_shape), 'int32')
	input_4D_time = T.reshape(input_time, new_shape, ndim=4)
	# downsample mini-batch of videos in time
	op = DownsampleFactorMax((1,ds[0]), ignore_border)
	outtime = op(input_4D_time)
	# restore to original shape (xxx, rows, cols, time)
	outshape = T.join(0, input_time.shape[:-2], outtime.shape[-2:])
	shufl = (list(range(vid_dim-4)) + [vid_dim-1] + list(range(vid_dim-4,vid_dim-1)))
	#shufl = (list(range(vid_dim-3)) + [vid_dim-1]+[vid_dim-3]+[vid_dim-2])
	return T.reshape(outtime, outshape, ndim=input.ndim).dimshuffle(shufl)
