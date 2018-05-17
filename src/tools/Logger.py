from collections import defaultdict
from constants import LOG_DIR
import datetime
import json


class Logger :
	def __init__(self, keys):
		self.logs = defaultdict([])
		self.keys = keys

	def log(self, keys, values):
		"""

		:param keys: list
		:param values: list
		"""
		for i in range(len(keys)) :
			self.logs[keys[i]] = values[i]

	def store(self, log_dir=LOG_DIR):

		with open(log_filen, 'w') as outfile:
			data = json.dumps(self.logs, indent=4, sort_keys=True)
			outfile.write(data)