import json
import autograd as a_grad

from Data import PoincareData
from constants import BURN_IN_EPOCHS, BURN_IN_RATE, MAX_RAND
from poincare_math import *
from tools.Logger import Logger
from utils import log_setup


class PoincareModel:
	def __init__(self, model_parameters, data_parameters):
		self.model_parameters = model_parameters
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
		self.dicts = self.data.dicts()


	def loss_fun(self, matrix, with_reg=True):
		u_vec = matrix[0]
		v_vecs = matrix[1:]
		nb_v_vecs = v_vecs.shape[0]

		poincare_dists = grad_np.array([poincare_dist(u_vec, v_vecs[i])
		                                for i in range(nb_v_vecs)])
		exp_neg_dists = grad_np.exp(-poincare_dists)
		reg_term = self.l2_reg * grad_np.linalg.norm(v_vecs[0]) ** 2
		if not with_reg :
			reg_term = 0.
		return - grad_np.log(exp_neg_dists[0] / (exp_neg_dists.sum())) + reg_term

	def compute_grad(self, sample, mod=grad_np):
		u_vec = self.theta[sample.u_id]
		v_vec = self.theta[sample.v_id]
		neigh_vecs = [self.theta[v_prime] for v_prime in sample.neigh_u_ids]
		matrix = mod.array([u_vec] + [v_vec] + neigh_vecs)

		grads = a_grad.grad(self.loss_fun)(matrix)
		idxs = [sample.u_id, sample.v_id] + sample.neigh_u_ids
		return {"grad": grads, "idxs": idxs}

	def update_parameters(self, gradient_data, learning_rate):
		idxs = gradient_data["idxs"]
		gradients = gradient_data["grad"]
		new_theta = self.theta[idxs] - learning_rate * gradients

		self.theta[idxs] = matrix_poincare_proj(new_theta)

	@staticmethod
	def initialize_theta(n, p, max_rand=MAX_RAND):
		return grad_np.array([[grad_np.random.uniform(- max_rand, max_rand)
		                  for _ in range(p)] for _ in range(n)])

	def train(self, epochs, learning_rate):
		print("{:15}|{:20}|{:20}".format("Epoch", "Loss", "Precision score"))
		for epoch in range(epochs):
			samples = self.data.batches(self.nb_neg_samples)
			self.print_log_performance(samples, epoch)
			for sample in samples:
				gradient = self.compute_grad(sample)
				self.update_parameters(gradient, learning_rate)

	def run(self, save=True):
		print("Burn-in ... ")
		if self.burn_in:
			self.train(epochs=BURN_IN_EPOCHS, learning_rate=BURN_IN_RATE)
		print("Learning ... ")
		self.train(epochs=self.epochs, learning_rate=self.learning_rate)
		if save:
			self.save_all()

	def print_log_performance(self, batch, epoch):
		loss = self.compute_loss(theta=self.theta,
		                         batch=self.data.loss_batch(self.nb_neg_samples))

		# reg = self.regularizer_loss([i for i in range(self.theta.shape[0])])
		precision = self.score(batch)

		print("{:15}|{:20}|{:20}".format(epoch, loss, precision))
		self.logger.log(["loss", "epoch", "precision"],
		                [loss, epoch, precision])

	def compute_individual_loss(self, theta, sample, mod):
		u_vec = theta[sample.u_id]
		v_vec = theta[sample.v_id]
		neigh_vecs = [theta[v_prime] for v_prime in sample.neigh_u_ids]
		matrix = mod.array([u_vec] + [v_vec] + neigh_vecs)
		return self.loss_fun(matrix, with_reg=False)

	def compute_loss(self, theta, batch, mod=grad_np):
		"""
		Computes the average loss over the batch
		:param batch: list of elements of class batch
		:return: (float) the average loss over the batch
		"""
		individual_losses = mod.array([self.compute_individual_loss(theta=theta,
		                                                            sample=sample,
		                                                            mod=mod)
		                               for sample in batch])

		return mod.sum(individual_losses) / individual_losses.shape[0]

	def save_model(self):
		filename = self.log_dir + "embeddings.out"
		grad_np.savetxt(filename, self.theta, delimiter=',')

		json_data = json.dumps(self.dicts, indent=4, sort_keys=True)
		filename = self.log_dir + "dicts.json"
		with open(filename, 'w') as outfile:
			data = json.dumps(json_data, indent=4, sort_keys=True)
			outfile.write(data)

		json_data = json.dumps(self.model_parameters, indent=4, sort_keys=True)
		filename = self.log_dir + "model_parameters.json"
		with open(filename, 'w') as outfile:
			data = json.dumps(json_data, indent=4, sort_keys=True)
			outfile.write(data)

	def save_all(self):
		self.save_model()
		self.logger.save()

	def load(self, model_dir, data_dir):
		pass

	def predict_sample(self, u, v):
		# TODO : check the prediction method
		dist = poincare_dist(self.theta[u], self.theta[v])
		if dist > 0.5:
			return 0
		return 1

	def predict(self, data):
		"""

		:param data: a list of pairs
		:return: a list of int : the predictions
		"""
		return [self.predict_sample(sample["u_id"], sample["v_id"]) for sample in
		        data]

	def score(self, data):
		"""
		A list of tuples (u_id, v_id, neigh_ids)
		:param data:
		:return: Mean accuracy for link prediction
		"""
		predictions = self.predict(data)
		return float(sum(predictions)) / len(predictions)



# def check_gradient(self, gradients, batch):
# 	true_gradient = self.compute_true_euc_grad(batch, self.theta)
# 	errors = [np.linalg.norm(gradients[str(i)] - true_gradient[i])
# 	          for i in range(true_gradient.shape[0])]
# 	return sum(errors) / len(errors)

# def compute_gradient(self, batch):
# 	riemann_grads = self.compute_riemann_gradient(batch)
# 	grads = self.compute_reg_grad(batch)
# 	return sum_grads(riemann_grads, grads)
#
# def compute_reg_grad(self, batch):
# 	unique_indices = self.batch_unique_indices(batch)
# 	# sub_theta = compute_sub_theta(self.theta, unique_indices)
# 	grads = defaultdict(partial(np.zeros, self.p))
# 	for idx in unique_indices:
# 		grads[idx] += self.l2_reg * self.theta[idx]
# 	return grads

# def compute_riemann_gradient(self, batch):
# 	"""
#
# 	:param batch: a list of (u_idx, v_idx, [v_neigh])
# 	:return: A dictionary, with indexes as keys and grads as vals
# 	"""
#
# 	grads = defaultdict(partial(np.zeros, self.p))
# 	# We first add the u term which is always present
# 	for sample in batch:
# 		grads = self.compute_riemann_grad_sample(u_id=sample["u_id"],
# 		                                         v_id=sample["v_id"],
# 		                                         neigh_u_ids=sample[
# 			                                         "neigh_u_ids"],
# 		                                         grads=grads)
# 	return grads
#
# def compute_riemann_grad_sample(self, u_id, v_id, neigh_u_ids, grads):
#
# 	# Compute (u, v) grad
# 	uv_grad = - d_poincare_dist(self.theta[u_id], self.theta[v_id])
# 	grads[str(u_id)] += uv_grad
# 	grads[str(v_id)] += uv_grad
#
# 	# Compute (u, N(u)) grads
# 	for v_prime_id in neigh_u_ids:
# 		uv_prime_grad = + d_poincare_dist(self.theta[u_id],
# 		                                  self.theta[v_prime_id])
# 		grad_coef = compute_poincare_coeff(u_id, v_prime_id, neigh_u_ids,
# 		                                   self.theta)
# 		grads[str(u_id)] += grad_coef * uv_prime_grad
# 		grads[str(v_id)] += grad_coef * uv_prime_grad
# 	return grads
# def update_parameters(self, riemman_gradient, learning_rate):
# 	for idx, grad in riemman_gradient.items():
# 		self.theta[int(idx)] = poincare_projection(self.theta[int(idx)] -
# 		                                           learning_rate * grad)

# @staticmethod
# def compute_unique_indices(u_idxs, v_idxs, neg_idxs):
# 	return list(set(u_idxs + v_idxs + neg_idxs))

# def batch_unique_indices(self, batch):
# 	u_idxs = [sample.u_id for sample in batch]
# 	v_idxs = [sample.v_id for sample in batch]
# 	neg_idxs = merge([sample.neigh_u_ids for sample in batch])
# 	return self.compute_unique_indices(u_idxs=u_idxs,
# 	                                   v_idxs=v_idxs,
# 	                                   neg_idxs=neg_idxs)




#
