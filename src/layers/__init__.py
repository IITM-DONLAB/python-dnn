
class nnetLayer(object):
	def __init__(self, *args, **kwargs):
		self.input = None
		self.W = None
		self.b = None
		self.delta_W = None
		self.delta_b = None 