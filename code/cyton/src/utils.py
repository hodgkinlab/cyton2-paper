import os
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

def optimal_bin(n, l):
	"""
	Calculate optimal bin number for a histogram.

	:param n: (int): number of samples
	:param l: (list): data    
	"""
	# R = max(l) - min(l)  # find range in data
	# std = np.std(l)  # compute standard deviation
	# return int(R * n ** (1. / 3.) / (3.49 * std))
	
	# Freedman–Diaconis rule
	iqr = np.subtract(*np.percentile(l, [75, 25]))
	h = 2. * iqr / (n**(1./3.))
	bins = (max(l) - min(l))/h
	return int(bins)

def ecdf(x):
	"""
	Calculate empirical cumulative distribution.

	:param x: (list) data
	:return: (tuple) x and y
	"""
	xs = np.sort(x)
	ys = np.arange(1, len(xs)+1)/float(len(xs))
	return xs, ys

def conf_iterval(l, rgs):
	alpha = (100. - rgs)/2.
	low = np.percentile(l, alpha, interpolation='nearest', axis=0)
	high = np.percentile(l, rgs+alpha, interpolation='nearest', axis=0)
	return (low, high)

def filter_hidden_and_sort(folder_path):
	""" filter hidden files and sort in alphabetical order

	This is unnecessary if you're absolutely sure that there are no other unwanted files in the folder. However,
	I often realised that when you copy over from Ubuntu to Mac or vice versa, the system automatically creates
	'.DS_Store' file.

	:param folder_path: (string) path to the data folder
	:return: (list) alphabetically sorted list of data files in the input folder
	"""
	return sorted(list(filter(lambda fname: not fname.startswith("."), os.listdir(folder_path))))

def remove_empty(l):
	""" recursively remove empty array from nested array
	:param l: (list) nested list with empty list(s)
	:return: (list)
	"""
	return list(filter(lambda x: not isinstance(x, (str, list, list)) or x, (remove_empty(x) if isinstance(x, (list, list)) else x for x in l)))


def set_share_axes(axs, target=None, sharex=False, sharey=False):
	if target is None:
		target = axs.flat[0]
	# Manage share using grouper objects
	for ax in axs.flat:
		if sharex:
			target._shared_x_axes.join(target, ax)
		if sharey:
			target._shared_y_axes.join(target, ax)
	# Turn off x tick labels and offset text for all but the bottom row
	if sharex and axs.ndim > 1:
		for ax in axs[:-1,:].flat:
			ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
			ax.xaxis.offsetText.set_visible(False)
	# Turn off y tick labels and offset text for all but the left most column
	if sharey and axs.ndim > 1:
		for ax in axs[:,1:].flat:
			ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
			ax.yaxis.offsetText.set_visible(False)
