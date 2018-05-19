from collections import defaultdict
import json


class Logger :
	def __init__(self, logdir, keys=None):
		self.logs = defaultdict(list)
		self.keys = keys
		self.logdir = logdir

	def log(self, keys, values):
		"""

		:param keys: list
		:param values: list
		"""
		for i in range(len(keys)) :
			self.logs[keys[i]].append(values[i])

	def save(self):
		filename = self.logdir + "logs.json"
		json_data = json.dumps(dict(self.logs), indent=4)
		with open(filename, 'w') as outfile:
			data = json.dumps(json_data, indent=4, sort_keys=True)
			outfile.write(data)