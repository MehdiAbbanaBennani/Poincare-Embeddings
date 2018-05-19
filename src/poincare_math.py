import numpy

from constants import EPSILON


def poincare_dist(u, v, mod=numpy) :
	if mod.linalg.norm(u - v) != 0 :
		num = mod.linalg.norm(u - v) ** 2
		den = (1 - mod.linalg.norm(u) ** 2) * (1 - mod.linalg.norm(v) ** 2)
		print(u, v)
		return mod.arccosh(1 + 2 * num / den)
	else:
		return 0


def poincare_projection(theta, epsilon=EPSILON, md=numpy):
	if md.linalg.norm(theta) >= 1 :
		return theta / md.linalg.norm(theta) - epsilon
	return theta


# def d_poincare_dist(theta, x, md=numpy):
# 	"""
# 	Partial derivative of the poincare distance
#
# 	:return: an array of dimension p
# 	"""
# 	if md.linalg.norm(x- theta) != 0 :
# 		beta = 1 - md.linalg.norm(x) ** 2 # (1)
# 		alpha = 1 - md.linalg.norm(theta) ** 2 # (1)
# 		gamma = 1 + 2 /(alpha * beta) * md.linalg.norm(theta - x)**2 # (1)
# 		left_coef = 4 / (beta * md.sqrt(gamma ** 2 - 1)) # (1)
#
# 		left_num = md.linalg.norm(x - theta) ** 2 + 1 - md.linalg.norm(theta) ** 2
# 		right_coef = left_num / alpha ** 2 * theta - x / alpha
# 		return left_coef * right_coef
#
# 	return np.zeros(np.shape(x))
#
#
# def compute_poincare_coeff(u_id, v_prime_id, neigh_u_ids, theta, md=numpy):
# 	"""
# 		Computes the S coefficient used in the gradient computation
# 	:param u_id: int
# 	:param v_prime_id: int
# 	:param neigh_u_ids: list of ints
# 	:param theta: (n,p) matrix
# 	:return: float
# 	"""
# 	num = md.exp(-poincare_dist(theta[u_id], theta[v_prime_id]))
# 	den = sum([md.exp(-poincare_dist(theta[u_id], theta[v_neigh_id]))
# 	           for v_neigh_id in neigh_u_ids])
# 	return num / den