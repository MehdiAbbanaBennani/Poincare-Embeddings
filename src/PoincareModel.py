import numpy as np
from numpy.linalg import norm
from numpy import log, exp
from poincare_math import poincare_dist, poincare_projection
from tools.Logger import Logger

from constants import MAX_RAND

class PoincareModel :
	def __init__(self, model_parameters, data_parameters):
		self.data = Data(data_parameters)
		self.learning_rate = model_parameters.learning_rate
		self.epochs = model_parameters.epochs
		self.Theta = self.initialize_theta(n=self.data.n, p=self.data.p)


		self.logger = Logger()

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

	def compute_riemman_gradient(self):


	def burn_in(self):
		pass



	def update_parameters(self, riemman_gradient):
		for idx, grad in riemman_gradient :
			self.theta[int(idx)] = poincare_projection(self.theta[int(idx)] -
			                                           self.learning_rate * grad)

	@staticmethod
	def initialize_theta(n, p, max_rand=MAX_RAND):
		return [[np.random.uniform(- max_rand, max_rand)
		         for _ in range(n)] for _ in range(p)]

	def learn(self):
		for epoch in range(self.epochs):
			for batch in self.Data.learn_batches():
				gradient = self.compute_gradient()
				self.theta[gradient.indices] -= self.learning_rate * gradient.gradient
				loss = self.compute_loss(batch)

				self.logger.log(["loss", "batch", "epoch"], [loss, batch, epoch])


	def save(self):


