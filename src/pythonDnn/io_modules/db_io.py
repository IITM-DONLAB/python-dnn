import dbm


def write_dataset(options):
	return dbwriter


class DbWriter:
	def __init__(self, options):
		self.db = dbm.open('websites', 'c')
	def close(self):
		self.db.close();
