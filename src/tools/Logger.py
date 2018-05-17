from collections import defaultdict
import json


class Logger :
	def __init__(self, logdir, keys=None):
		self.logs = defaultdict([])
		self.keys = keys
		self.logdir = logdir

	def log(self, keys, values):
		"""

		:param keys: list
		:param values: list
		"""
		for i in range(len(keys)) :
			self.logs[keys[i]] = values[i]

	def store(self):
		filename = self.logdir + "logs.json"
		with open(filename, 'w') as outfile:
			data = json.dumps(self.logs, indent=4, sort_keys=True)
			outfile.write(data)