import numpy as np


def sum_grads(grads_1, grads_2):
	"""
	Compute the sum of grads_1 and grads_2
	:param grads_1: dictionary of gradients, where the key is the index and the value the grad
	:param grads_2: dictionary of gradients, where the key is the index and the value the grad
	:return: a dictionary
	"""
	for idx, grad in grads_2.items():
		grads_1[idx] += grad
	return grads_1


def compute_sub_theta(theta, idx):
	"""

	:param theta: a np matrix
	:param idx: a list of indexes
	:return: a np matrix
	"""
	return theta[idx]


def matrix_norm(theta, idx):
	"""

	:param Theta: (n, d) matrix
	idx : list of indices
	:return: a float
	"""
	sub_theta = compute_sub_theta(theta, idx)
	return np.sum(np.multiply(sub_theta, sub_theta))