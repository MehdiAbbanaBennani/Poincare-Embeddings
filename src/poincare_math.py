from numpy import arccosh, sqrt, exp
from numpy.linalg import norm
import numpy as np

from constants import EPSILON


def poincare_dist(u, v) :
	num = norm(u - v) ** 2
	den = (1 - norm(u) ** 2) * (1 - norm(v) ** 2)
	return arccosh(1 + 2 * num / den)


def poincare_projection(theta, epsilon=EPSILON):
	if norm(theta) >= 1 :
		return theta / norm(theta) - epsilon
	return theta


def d_poincare_dist(theta, x):
	"""
	Partial derivative of the poincare distance

	:return: an array of dimension p
	"""
	if norm(x- theta) != 0 :
		beta = 1 - norm(x) ** 2 # (1)
		alpha = 1 - norm(theta) ** 2 # (1)
		gamma = 1 + 2 /(alpha * beta) * norm(theta - x)**2 # (1)
		left_coef = 4 / (beta * sqrt(gamma ** 2 - 1)) # (1)

		left_num = norm(x - theta) ** 2 + 1 - norm(theta) ** 2
		right_coef = left_num / alpha ** 2 * theta - x / alpha
		return left_coef * right_coef

	return np.zeros(np.shape(x))

def matrix_norm(theta, idx):
	"""

	:param Theta: (n, d) matrix
	idx : list of indices
	:return: float
	"""
	sub_theta = theta[idx]
	return np.sum(np.multiply(sub_theta, sub_theta))


def compute_poincare_coeff(u_id, v_prime_id, neigh_u_ids, theta):
	"""
		Computes the S coefficient used in the gradient computation
	:param u_id: int
	:param v_prime_id: int
	:param neigh_u_ids: list of ints
	:param theta: (n,p) matrix
	:return: float
	"""
	num = exp(-poincare_dist(theta[u_id], theta[v_prime_id]))
	den = sum([exp(-poincare_dist(theta[u_id], theta[v_neigh_id]))
	           for v_neigh_id in neigh_u_ids])
	return num / den