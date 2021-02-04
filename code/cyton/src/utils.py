import numpy as np
import scipy.stats as sps

def norm_pdf(x, mu, sig):
	return sps.norm.pdf(x, mu, sig)
def norm_cdf(x, mu, sig):
	return sps.norm.cdf(x, mu, sig)

def truncnorm_pdf(x, mu, sig):
	return sps.truncnorm.pdf(x, a=(0-mu)/sig, b=np.inf, loc=mu, scale=sig)
def truncnorm_cdf(x, mu, sig):
	return sps.truncnorm.cdf(x, a=(0-mu)/sig, b=np.inf, loc=mu, scale=sig)

def lognorm_pdf(x, m, s):
	return sps.lognorm.pdf(x, s, scale=m)
def lognorm_cdf(x, m, s):
	return sps.lognorm.cdf(x, s, scale=m)

def lognorm_statistics(m, s, return_std=False):
	"""
	Calculates statistics of the mean and variance of Lognormal distribution.

	:param m: (float) the median
	:param s: (float) the standard deviation
	:param std: (float) return standard deviation if True
	"""
	mean = np.exp(np.log(m) + s**2/2)
	var = np.exp(2*np.log(m) + 2*s**2) - np.exp(2*np.log(m) + s**2)
	if return_std:
		return mean, np.sqrt(var)
	return mean, var

def ecdf(x):
	"""
	Calculate empirical cumulative distribution.

	:param x: (list) data
	:return: (tuple) x and y
	"""
	n = len(x)
	xs = np.sort(x)
	ys = np.arange(1, n+1)/float(n)
	return xs, ys

def conf_iterval(l, rgs):
	alpha = (100. - rgs)/2.
	low = np.percentile(l, alpha, interpolation='nearest', axis=0)
	high = np.percentile(l, rgs+alpha, interpolation='nearest', axis=0)
	return (low, high)

def remove_empty(l):
	""" recursively remove empty array from nested array
	:param l: (list) nested list with empty list(s)
	:return: (list)
	"""
	return list(filter(lambda x: not isinstance(x, (str, list, list)) or x, (remove_empty(x) if isinstance(x, (list, list)) else x for x in l)))
