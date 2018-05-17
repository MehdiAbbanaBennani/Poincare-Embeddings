from numpy import exp, log

from constants import BURN_IN_EPOCHS, BURN_IN_RATE, MAX_RAND, LOG_DIR
from poincare_math import *
from tools.Logger import Logger
import pickle
from utils import log_setup


class PoincareModel:
	def __init__(self, model_parameters, data_parameters):
		self.data = Data(data_parameters)
		self.learning_rate = model_parameters.learning_rate
		self.epochs = model_parameters.epochs

		self.theta = self.initialize_theta(n=self.data.n, p=self.data.p)

		self.burn_in = model_parameters.burn_in
		self.l2_reg = model_parameters.l2_reg

		self.log_dir = log_setup()
		self.logger = Logger(self.log_dir)



	def compute_loss(self, batch):
		"""
		Computes the average loss over the batch
		:param batch: list of elements of class batch
		:return: (float) the average loss over the batch
		"""
		individual_losses = [self.compute_individual_loss(u=batch[i].u,
		                                                  v=batch[i].v,
		                                                  neg_samples=batch[
			                                                  i].neg_samples)
		                     for i in range(len(batch))]
		reg_loss = self.regularizer_loss()
		return np.mean(individual_losses) + reg_loss

	def compute_individual_loss(self, u, v, neg_samples):
		num = exp(- poincare_dist(u, v))
		den = sum([exp(- poincare_dist(u, v_prime)) for v_prime in neg_samples])
		return log(num / den)

	def compute_euclidian_distances(self):
		self.euclidian_distances = np.array([[norm((u - v))
		                                      for u in self.data.unique_vectors]
		                                     for v in self.data.unique_vectors])

	def compute_riemman_gradient(self, batch):
		"""
		
		:param batch: 
		:return: A dictionary, with indexes as keys and grads as vals
		"""

		grads = {}
		# We first add the u term which is always present
		for sample in batch:
			grads = self.compute_riemman_grad_sample(u_id=sample.u_id,
			                                         v_id=sample.v_id,
			                                         neigh_u_ids=sample.neigh_u_ids,
			                                         grads=grads)
		return grads

	def compute_riemman_grad_sample(self, u_id, v_id, neigh_u_ids, grads):
		# grads = defaultdict([])

		# Compute (u, v) grad
		uv_grad = - d_poincare_dist(self.theta[u_id],
		                            self.theta[v_id])
		grads[str(u_id)] = uv_grad
		grads[str(v_id)] = uv_grad

		# Compute (u, N(u)) grads
		for v_prime_id in neigh_u_ids:
			uv_prime_grad = d_poincare_dist(self.theta[u_id],
			                                self.theta[v_prime_id])
			grads[str(u_id)] = uv_prime_grad
			grads[str(v_id)] = uv_prime_grad

		return grads

	def regularizer_loss(self):
		return self.L2_loss * matrix_norm(theta=self.theta)

	def update_parameters(self, riemman_gradient, learning_rate):
		for idx, grad in riemman_gradient:
			self.theta[int(idx)] = poincare_projection(self.theta[int(idx)] -
			                                           learning_rate * grad)

	@staticmethod
	def initialize_theta(n, p, max_rand=MAX_RAND):
		return [[np.random.uniform(- max_rand, max_rand)
		         for _ in range(p)] for _ in range(n)]

	def run(self):
		if self.burn_in:
			self.train(epochs=BURN_IN_EPOCHS, learning_rate=BURN_IN_RATE)
		self.train(epochs=self.epochs, learning_rate=self.learning_rate)

	def train(self, epochs, learning_rate):
		for epoch in range(epochs):
			for batch in self.Data.learn_batches():
				riemman_gradient = self.compute_riemman_gradient()
				self.update_parameters(riemman_gradient, learning_rate)
				loss = self.compute_loss(batch)

				self.logger.log(["loss", "batch", "epoch"], [loss, batch, epoch])

	def save(self):
		filename = self.log_dir + "model.pkl"
		with open(filename, 'w') as f :
			pickle.dump(self.theta, f)