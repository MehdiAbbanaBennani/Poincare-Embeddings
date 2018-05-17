from numpy import arccosh, sqrt
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
	:return:
	"""
	beta = 1 - norm(x) ** 2
	alpha = 1 - norm(theta) ** 2
	gamma = 1 + 2 /(alpha * beta) * norm(theta - x)**2
	left_coef = 4 / (beta * sqrt(gamma ** 2 - 1))

	left_num = norm(x - theta) ** 2 + 1 - norm(theta) ** 2 / alpha ** 2
	right_coef = left_num / alpha ** 2 * theta - x / alpha
	return left_coef * right_coef


def matrix_norm(theta):
	return np.sum(np.multiply(theta))
