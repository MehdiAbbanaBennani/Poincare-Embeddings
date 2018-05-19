import json

import numpy as np

from poincare_math import poincare_dist


class Embeddings:
	def __init__(self, directory):
		self.word2index, self.index2word = self.load_dicts(directory)
		self.embeddings = self.load_embeddings(directory)
		self.word_list = list(self.word2index.keys())

	@staticmethod
	def load_embeddings(directory):
		filename = directory + "embeddings.out"
		return np.loadtxt(filename, delimiter=",")

	@staticmethod
	def load_dicts(directory):
		filename = directory + "dicts.json"
		with open(filename) as f:
			data_str = json.load(f)
			data = json.loads(data_str)
		return data["w_dict"], data["i_dict"]

	def dist(self, word_1, word_2, type):
		"""

		:param word_1: string
		:param word_2: string
		:param type: either 'euclidian' or 'poincare'
		:return:
		"""
		id_1 = self.word2index[word_1]
		id_2 = self.word2index[word_2]
		if type == 'poincare':
			return poincare_dist(self.embeddings[id_1], self.embeddings[id_2])
		if type == "euclidian":
			return np.linalg.norm(self.embeddings[id_1] - self.embeddings[id_2])

	def norm(self, word):
		id = self.word2index[word]
		return np.linalg.norm(self.embeddings[id])

	def search_word(self, word):
		return [s for s in self.word_list if word in s]
