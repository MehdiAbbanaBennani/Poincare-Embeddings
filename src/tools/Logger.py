from collections import defaultdict
from constants import LOG_DIR
import datetime
import json


class Logger :
	def __init__(self, keys):
		self.logs = defaultdict([])
		self.keys = keys

	def log(self, key, value):
		self.logs[key] = value

	def store(self, log_dir=LOG_DIR):
		now = datetime.datetime.now().strftime("%b:%d:%Y:%H:%M:%S")
		log_filen = LOG_DIR + "run_logs:" + now + ".json"
		with open(log_filen, 'w') as outfile:
			data = json.dumps(self.logs, indent=4, sort_keys=True)
			outfile.write(data)