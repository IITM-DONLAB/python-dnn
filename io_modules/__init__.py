
def setLogger(level="INFO",stderr=True,name=None,logFile='python-dnn.log'):
	import logging,sys

	#get Logger
	logger = logging.getLogger(name)

	if stderr:
		# Send the logs to stderr
		stream_handler = logging.StreamHandler()
		# Format the log output and include the log level's name and the time it was generated
		formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
		# Use that Formatter on handler
		stream_handler.setFormatter(formatter)
		# Add the handler to it
		logger.addHandler(stream_handler)
	else :
		# Set up a log file
		file_handler = logging.FileHandler(logFile)
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)

	# Set the level which determines what you see
	if level == "INFO":
		logger.setLevel(logging.INFO)
	elif level=="DEBUG":
		logger.setLevel(logging.DEBUG)
	else:
		logger.setLevel(logging.ERROR)