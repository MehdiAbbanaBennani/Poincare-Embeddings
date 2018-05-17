import io
import os
import numpy as np
import scipy
from scipy import linalg
import sys
from smart_open import smart_open
import csv
from collections import defaultdict


# PATH_TO_DATA = os.path.join(os.getcwd(),'data')
# os.listdir(PATH_TO_DATA)
# wordnet_mammal_file = os.path.join(PATH_TO_DATA, 'wordnet_mammal_hypernyms.tsv')


class PoincareData():
    def __init__(self,fname, nmax, verbose = False, doublon_tol = False):
        '''
        fname: name of the tsv file
        nmax: number of extracted lines in the fname file
        '''
        l = self.load_file(fname,nmax, verbose)
        self.build_vocab(l, verbose, doublon_tol)

    def load_file(self,fname, nmax, verbose):
        '''
        Parsing of the TSV file
        '''
        entire = True
        with open(fname,'r') as f:
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

    def build_vocab(self,loaded_file, verbose, doublon_tol = False):
        '''
        Given a loaded_file, build the index2word, word2index, the vocab,
        node relations
        param:
            - vocab: occurence of the different words (defaultdict)
            - index2word: index associated to a word (list)
            - all_relations : list of tuples containing the interactions (list)
            - word2index: (dict)
            - node_relations: mapping from node index to its related node indices
        '''
        self.vocab = defaultdict(lambda: 0)
        self.index2word = [] # position du mot dans la liste est l'index
        self.all_relations = []
        self.word2index = {}
        self.node_relations = defaultdict(set)
        doublon = 0
        #self.wordfrequency = {}
        for relation in loaded_file:
            if relation[0] == relation[1]:
                doublon +=1
            if len(relation) != 2:
                raise ValueError('Relation pair "%s" should be a pair !'
                % repr(relation))
            if (doublon_tol == True):
                for w in relation:
                    if w in self.vocab:
                        self.vocab[w] +=1
                    else:
                        # new word detected
                        self.word2index[w] = len(self.index2word)
                        # we give the new word its own index
                        self.index2word.append(w) # new word in the list
                        self.vocab[w] = 1 # new key in the vocab dictionary

                node1,node2 = relation
                node1_index, node2_index = (self.word2index[node1],
                                            self.word2index[node2])
                self.node_relations[node1_index].add(node2_index)
                self.all_relations.append((node1_index, node2_index))
            else:
                if relation[0] != relation[1]:
                    for w in relation:
                        if w in self.vocab:
                            self.vocab[w] +=1
                        else:
                            # new word detected
                            self.word2index[w] = len(self.index2word)
                            self.index2word.append(w)
                            self.vocab[w] = 1

                    node1,node2 = relation
                    node1_index, node2_index = (self.word2index[node1],
                                                self.word2index[node2])
                    self.node_relations[node1_index].add(node2_index)
                    self.all_relations.append((node1_index, node2_index))
        if verbose:
            print('Vocabulary Build !')
            print(str(doublon) + ' doublons was found')

		def batches():