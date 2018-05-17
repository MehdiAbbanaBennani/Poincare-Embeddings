from numpy import arccosh
from numpy.linalg import norm


def poincare_dist(u, v) :
	num = norm(u - v) ** 2
	den = (1 - norm(u) ** 2) * (1 - norm(v) ** 2)
	return arccosh(1 + 2 * num / den)