import numpy as np
from collections import defaultdict
from utils import dotdict

class PoincareData():
	def __init__(self, fname, nmax, verbose=False, doublon_tol=False):
		'''
		fname: name of the tsv file
		nmax: number of extracted lines in the fname file
		'''
		l = self.load_file(fname, nmax, verbose)
		self.build_vocab(l, verbose, doublon_tol)
		self.no_neigh()
		self.n_unique_words = len(self.word2index)

	def load_file(self, fname, nmax, verbose):
		'''
		Parsing of the TSV file
		'''
		entire = True
		with open(fname, 'r') as f:
			l = []
			for i, line in enumerate(f):
				if i >= nmax:
					entire = False
					break
				line = line.strip().split('\t')
				l.append(tuple(line))
		if verbose:
			print('Entire Parsing: ' + str(entire))
		return l

	def batches(self, nb_neg_samples):
		batch = []
		for el in self.all_relations:
			id1, id2 = el
			negative_samples = self.negative_samples(id1, nb_neg_samples)
			sample_dict = {"u_id" : id1,
			               "v_id" : id2,
			               "neigh_u_ids" : negative_samples}
			batch.append([dotdict(sample_dict)])
		return batch

	def loss_batch(self, nb_neg_samples):
		batch = []
		for el in self.all_relations:
			id1, id2 = el
			negative_samples = self.negative_samples(id1, nb_neg_samples)
			sample_dict = {"u_id" : id1,
			               "v_id" : id2,
			               "neigh_u_ids" : negative_samples}
			batch.append(dotdict(sample_dict))
		return batch

	def no_neigh(self):
		self.not_neigh = {}
		index_max = len(self.index2word)
		for i in range(index_max):
			self.not_neigh[i] = [j for j in list(range(index_max)) if
			                     j not in self.node_relations[i]]

	def negative_samples(self, idx, N):
		'''
		Return N negative words idx for the word of index idx based on the frequency of the different words
		'''
		dico_freq_not_neigh = {}
		for el in self.not_neigh[idx]:
			dico_freq_not_neigh[el] = self.wordfrequency[idx]
		freq_tot = sum(dico_freq_not_neigh.values())
		keys = list(dico_freq_not_neigh.keys())
		probs = list(dico_freq_not_neigh.values())
		probs = [el / freq_tot for el in probs]  # normalized vector
		return list(np.random.choice(keys, N, p=probs, replace=False)+ [idx])

	def build_vocab(self, loaded_file, verbose, doublon_tol=False):
		'''
		Given a loaded_file, build the index2word, word2index, the vocab, node relations
		param:
				- vocab: occurence of the different words (defaultdict)
				- index2word: index associated to a word (list)
				- all_relations : list of tuples containing the interactions (list)
				- word2index: (dict)
				- node_relations: mapping from node index to its related node indices
		'''
		self.vocab = defaultdict(lambda: 0)
		self.index2word = []  # position du mot dans la liste est l'index
		self.all_relations = []
		self.word2index = {}
		self.node_relations = defaultdict(set)
		doublon = 0
		self.wordfrequency = {}
		for relation in loaded_file:
			if relation[0] == relation[1]:
				doublon += 1
			if len(relation) != 2:
				raise ValueError(
					'Relation pair "%s" should be a pair !' % repr(relation))
			if (doublon_tol == True):
				for w in relation:
					if w in self.vocab:
						self.vocab[w] += 1
					else:
						# new word detected
						self.word2index[w] = len(
							self.index2word)  # we give the new word its own index
						self.index2word.append(w)  # new word in the list
						self.vocab[w] = 1  # new key in the vocab dictionary

				node1, node2 = relation
				node1_index, node2_index = self.word2index[node1], self.word2index[
					node2]
				self.node_relations[node1_index].add(node2_index)
				self.all_relations.append((node1_index, node2_index))
			else:
				if relation[0] != relation[1]:
					for w in relation:
						if w in self.vocab:
							self.vocab[w] += 1
						else:
							# new word detected
							self.word2index[w] = len(self.index2word)
							self.index2word.append(w)
							self.vocab[w] = 1

					node1, node2 = relation
					node1_index, node2_index = self.word2index[node1], self.word2index[
						node2]
					self.node_relations[node1_index].add(node2_index)
					self.all_relations.append((node1_index, node2_index))
		if verbose:
			print('Vocabulary Build !')
			print(str(doublon) + ' doublons was found')
		freq_tot = sum(self.vocab.values())
		for key, value in self.vocab.items():
			self.wordfrequency[self.word2index[key]] = value / freq_tot
