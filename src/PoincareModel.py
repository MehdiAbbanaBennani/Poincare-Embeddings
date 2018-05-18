import pickle
from collections import defaultdict
from functools import partial

from numpy import log
import autograd.numpy as grad_np

from Data import PoincareData
from constants import BURN_IN_EPOCHS, BURN_IN_RATE, MAX_RAND
from poincare_math import *
from tools.Logger import Logger
from utils import log_setup, merge
from math_tools import *


class PoincareModel:
	def __init__(self, model_parameters, data_parameters):
		self.data = PoincareData(data_parameters["filename"],
		                         data_parameters["nmax"])
		self.learning_rate = model_parameters["learning_rate"]
		self.epochs = model_parameters["epochs"]
		self.p = model_parameters["p"]
		self.burn_in = model_parameters["burn_in"]
		self.l2_reg = model_parameters["l2_reg"]
		self.nb_neg_samples = model_parameters["nb_neg_samples"]

		self.theta = self.initialize_theta(n=self.data.n_unique_words, p=self.p)

		self.log_dir = log_setup()
		self.logger = Logger(self.log_dir)

	def compute_loss(self, batch):
		"""
		Computes the average loss over the batch
		:param batch: list of elements of class batch
		:return: (float) the average loss over the batch
		"""
		individual_losses = [self.compute_individual_loss(sample) for sample in
		                     batch]

		unique_indices = self.batch_unique_indices(batch)
		reg_loss = self.regularizer_loss(unique_indices)
		return np.mean(individual_losses) + reg_loss

	def batch_unique_indices(self, batch):
		u_idxs = [sample.u_id for sample in batch]
		v_idxs = [sample.v_id for sample in batch]
		neg_idxs = merge([sample.neigh_u_ids for sample in batch])
		return self.compute_unique_indices(u_idxs=u_idxs,
		                                   v_idxs=v_idxs,
		                                   neg_idxs=neg_idxs)

	@staticmethod
	def compute_unique_indices(u_idxs, v_idxs, neg_idxs):
		return list(set(u_idxs + v_idxs + neg_idxs))

	def compute_individual_loss(self, sample):
		num = exp(- poincare_dist(self.theta[sample.u_id],
		                          self.theta[sample.v_id]))
		den = sum([exp(- poincare_dist(self.theta[sample.u_id],
		                               self.theta[v_prime]))
		           for v_prime in sample.neigh_u_ids])
		return log(num / den)

	# def compute_euclidian_distances(self):
	# 	self.euclidian_distances = np.array([[norm((u - v))
	# 	                                      for u in self.data.unique_vectors]
	# 	                                     for v in self.data.unique_vectors])

	def compute_gradient(self, batch):
		riemann_grads = self.compute_riemann_gradient(batch)
		grads = self.compute_reg_grad(batch)
		return sum_grads(riemann_grads, grads)

	def compute_reg_grad(self, batch):
		unique_indices = self.batch_unique_indices(batch)
		# sub_theta = compute_sub_theta(self.theta, unique_indices)
		grads = defaultdict(partial(np.zeros, self.p))
		for idx in unique_indices :
			grads[idx] += self.l2_reg * self.theta[idx]
		return grads

	def compute_riemann_gradient(self, batch):
		"""
		
		:param batch: a list of (u_idx, v_idx, [v_neigh])
		:return: A dictionary, with indexes as keys and grads as vals
		"""

		grads = defaultdict(partial(np.zeros, self.p))
		# We first add the u term which is always present
		for sample in batch:
			grads = self.compute_riemann_grad_sample(u_id=sample["u_id"],
			                                         v_id=sample["v_id"],
			                                         neigh_u_ids=sample[
				                                         "neigh_u_ids"],
			                                         grads=grads)
		return grads

	def compute_riemann_grad_sample(self, u_id, v_id, neigh_u_ids, grads):

		# Compute (u, v) grad
		uv_grad = - d_poincare_dist(self.theta[u_id], self.theta[v_id])
		grads[str(u_id)] += uv_grad
		grads[str(v_id)] += uv_grad

		# Compute (u, N(u)) grads
		for v_prime_id in neigh_u_ids:
			uv_prime_grad = + d_poincare_dist(self.theta[u_id],
			                                  self.theta[v_prime_id])
			grad_coef = compute_poincare_coeff(u_id, v_prime_id, neigh_u_ids,
			                                   self.theta)
			grads[str(u_id)] += grad_coef * uv_prime_grad
			grads[str(v_id)] += grad_coef * uv_prime_grad
		return grads

	def regularizer_loss(self, idx):
		return self.l2_reg * matrix_norm(theta=self.theta, idx=idx)

	def update_parameters(self, riemman_gradient, learning_rate):
		for idx, grad in riemman_gradient.items():
			self.theta[int(idx)] = poincare_projection(self.theta[int(idx)] -
			                                           learning_rate * grad)

	@staticmethod
	def initialize_theta(n, p, max_rand=MAX_RAND):
		return np.array([[np.random.uniform(- max_rand, max_rand)
		                  for _ in range(p)] for _ in range(n)])

	def train(self, epochs, learning_rate):
		for epoch in range(epochs):
			batches = self.data.batches(self.nb_neg_samples)
			for batch in batches:
				# riemman_gradient = self.compute_riemann_gradient(batch)
				gradient = self.compute_gradient(batch)
				self.update_parameters(gradient, learning_rate)
			loss = self.compute_loss(self.data.loss_batch(self.nb_neg_samples))

			reg = self.regularizer_loss([i for i in range(self.theta.shape[0])])
			print("{:15}|{:20}|{:20}".format(epoch, loss, reg))
			self.logger.log(["loss", "batch", "epoch"], [loss, batch, epoch])

	def run(self, save=True):
		if self.burn_in:
			self.train(epochs=BURN_IN_EPOCHS, learning_rate=BURN_IN_RATE)
		self.train(epochs=self.epochs, learning_rate=self.learning_rate)
		if save:
			self.save_all()

	def save_model(self):
		filename = self.log_dir + "model.pkl"
		with open(filename, 'wb') as f:
			pickle.dump(self.theta, f)

	def save_all(self):
		self.save_model()
		self.logger.save()

	def load(self, model_dir, data_dir):
		pass

	@staticmethod
	def predict_sample(u, v):
		# TODO : check the prediction method
		dist = poincare_dist(u, v)
		if dist > 0.5:
			return 0
		return 1

	def predict(self, data):
		"""

		:param data: a list of pairs
		:return: a list of int : the predictions
		"""
		return [self.predict_sample(*sample) for sample in data]

	def score(self, data):
		"""
		A list of tuples (u_id, v_id, neigh_ids)
		:param data:
		:return: Mean accuracy for link prediction
		"""
		predictions = self.predict(data)
		return sum(predictions) / len(predictions)