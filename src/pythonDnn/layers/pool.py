import theano.tensor as T
from util.max_pool import max_pool_3d
	
class PoolLayer(object):
	""" Subsampling and pooling layer """
	def __init__(self, input, pool_shape, method="max"):
		"""
		method: "max", "avg", "L2", "L4", ...
		"""
		#self.__dict__.update(locals())
		#del self.self
		if len(pool_shape) == 0:
			self.output=input;
			return
		if method=="max":
			if len(pool_shape) == 3:
				max_pool = max_pool_3d;
			else:
				from theano.tensor.signal import downsample
				max_pool = downsample.max_pool_2d
			out = max_pool(input=input,ds=pool_shape,ignore_border=True)
		else:
			raise NotImplementedError()
		self.output = out
	