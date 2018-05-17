import numpy as np
from numpy.linalg import norm
from numpy import log, exp
from utils import poincare_dist

class PoincareModel :
	def __init__(self):
		self.data = Data(data)
		self.learning_rate = learning_rate

	def compute_loss(self, batch):
		"""
		Computes the average loss over the batch
		:param batch: list of elements of class batch
		:return: (float) the average loss over the batch
		"""
		individual_losses = [self.compute_individual_loss(u=batch[i].u,
		                                                  v=batch[i].v,
		                                                  neg_samples=batch[i].neg_samples)
		                     for i in range(len(batch))]
		return np.mean(individual_losses)

	def compute_individual_loss(self, u, v, neg_samples):
		num = exp(- poincare_dist(u, v))
		den = sum([exp(- poincare_dist(u, v_prime)) for v_prime in neg_samples])
		return log(num / den)

	def compute_euclidian_distances(self):
		self.euclidian_distances = np.array([[norm((u - v))
		                     for u in self.data.unique_vectors]
		                    for v in self.data.unique_vectors])

	def compute_norm(self):



