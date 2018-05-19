import autograd.numpy as grad_np

from constants import EPSILON


def poincare_dist(u, v, mod=grad_np) :
	if mod.linalg.norm(u) > 1 or mod.linalg.norm(v) > 1 :
		print(mod.linalg.norm(u), mod.linalg.norm(v))
	euc_dist = mod.linalg.norm(u - v)
	if euc_dist > EPSILON :
		u_norm = mod.linalg.norm(u)
		v_norm = mod.linalg.norm(v)
		return mod.arccosh(1 + 2 * (
				(euc_dist ** 2) / ((1 - u_norm ** 2) * (1 - v_norm ** 2))
		))
	else:
		return 0


def matrix_poincare_proj(theta, md=grad_np):
	return md.array([poincare_projection(theta[i])
	                 for i in range(theta.shape[0])])


def poincare_projection(theta, epsilon=EPSILON, md=grad_np):
	if md.linalg.norm(theta) >= 1 :
		normalized = theta / md.linalg.norm(theta)
		shifted = normalized - epsilon * md.sign(normalized)
		# if md.linalg.norm(shifted) > 1 :
		# 	print(md.linalg.norm(shifted))
		return shifted
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