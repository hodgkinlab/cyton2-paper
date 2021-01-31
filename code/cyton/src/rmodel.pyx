"""
Last edit: 21-November-2020

Reduced model of Cyton 1.5. Only care about activated cells (i.e. p=1 case).
"""
import numpy as np
cimport numpy as np
np.get_include()
from scipy.stats import lognorm, norm

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cimport cython
@cython.boundscheck(True)
@cython.wraparound(True)
@cython.nonecheck(True)
@cython.cdivision(True)
class Cyton15Model:
	def __init__(self, ht, n0, max_div, dt, nreps, logn=True):
		self.t0 = <DTYPE_t>(0.0)
		self.tf = <DTYPE_t>(max(ht) + dt)
		self.dt = <DTYPE_t>(dt)  									# time increment

		# declare time array
		self.times = np.arange(self.t0, self.tf, dt, dtype=DTYPE)
		self.nt = <unsigned int>(self.times.size)

		self.n0 = <DTYPE_t>n0  										# experiment initial cell number
		self.ht = ht  												# experiment harvested times
		self.nreps = nreps  										# experiment number of replicates

		self.exp_max_div = <unsigned int>max_div  					# observed maximum division number
		self.max_div = <unsigned int>10  							# theoretical maximum division number
		self.logn = logn

	def compute_pdf(self, times, mu, sig):
		if self.logn:
			return lognorm.pdf(times, sig, scale=mu)
		else:
			return norm.pdf(times, mu, sig)

	def compute_cdf(self, times, mu, sig):
		if self.logn:
			return lognorm.cdf(times, sig, scale=mu)
		else:
			return norm.cdf(times, mu, sig)

	def compute_sf(self, times, mu, sig):
		if self.logn:
			return lognorm.sf(times, sig, scale=mu)
		else:
			return norm.sf(times, mu, sig)

	def _storage(self):
		cdef np.ndarray[DTYPE_t, ndim=1] pdfDD = np.zeros(shape=self.nt, dtype=DTYPE)

		cdef np.ndarray[DTYPE_t, ndim=1] sfDiv = np.zeros(shape=self.nt, dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1] sfDie = np.zeros(shape=self.nt, dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1] sfDD = np.zeros(shape=self.nt, dtype=DTYPE)

		# declare 2 arrays for divided cells & destiny cells
		cdef np.ndarray[DTYPE_t, ndim=2] nDIV = np.zeros(shape=(self.max_div+1, self.nt), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=2] nDES = np.zeros(shape=(self.max_div+1, self.nt), dtype=DTYPE)

		# store number of live cells at all time per generations
		cdef np.ndarray[DTYPE_t, ndim=2] cells_gen = np.zeros(shape=(self.exp_max_div+1, self.nt), dtype=DTYPE)

		return pdfDD, sfDiv, sfDie, sfDD, nDIV, nDES, cells_gen

	# return sum of dividing and destiny cells in each generation
	def evaluate(self,
			DTYPE_t mDiv0, DTYPE_t sDiv0,  # time to first division
			DTYPE_t mDD, DTYPE_t sDD,      # division destiny
			DTYPE_t mDie, DTYPE_t sDie,    # stimulated death
			DTYPE_t m):                    # subsequent division time
		cdef np.ndarray[DTYPE_t, ndim=1] times
		times = self.times

		# create empty arrays
		pdfDD, sfDiv, sfDie, sfDD, nDIV, nDES, cells_gen = self._storage()

		# compute probability distribution
		pdfDD = self.compute_pdf(times, mDD, sDD)

		# compute survival functions (i.e. 1 - cdf)
		sfDiv = self.compute_sf(times, mDiv0, sDiv0)
		sfDie = self.compute_sf(times, mDie, sDie)
		sfDD = self.compute_sf(times, mDD, sDD)

		# calculate gen = 0 cells
		cdef DTYPE_t x, y
		nDIV[0,:] = self.n0 * sfDie * sfDiv * sfDD
		nDES[0,:] = self.n0 * sfDie * np.cumsum([x * y for x, y in zip(pdfDD, sfDiv)]) * self.dt
		cells_gen[0,:] = nDIV[0,:] + nDES[0,:]  # cells in generation 0

		# calculate gen > 0 cells
		cdef DTYPE_t core
		cdef unsigned int igen
		cdef np.ndarray[DTYPE_t, ndim=1] upp_cdfDiv
		cdef np.ndarray[DTYPE_t, ndim=1] low_cdfDiv
		cdef np.ndarray[DTYPE_t, ndim=1] difference
		for igen in range(1, self.max_div+1):
			core = <DTYPE_t>(2.**igen * self.n0)

			upp_cdfDiv = self.compute_cdf(times - <DTYPE_t>((igen - 1.)*m), mDiv0, sDiv0)
			low_cdfDiv = self.compute_cdf(times - <DTYPE_t>(igen*m), mDiv0, sDiv0)
			difference = upp_cdfDiv - low_cdfDiv

			nDIV[igen,:] = core * sfDie * sfDD * difference
			nDES[igen,:] = core * sfDie * np.cumsum([x * y for x, y in zip(pdfDD, difference)]) * self.dt

			if igen < self.exp_max_div:
				cells_gen[igen,:] = nDIV[igen,:] + nDES[igen,:]
			else:
				cells_gen[self.exp_max_div,:] += nDIV[igen,:] + nDES[igen,:]
		
		# extract number of live cells at harvested time points from 'cells_gen' array
		cdef DTYPE_t cell, ht
		cdef unsigned int itpt, irep, t_idx
		cdef list model = []
		for itpt, ht in enumerate(self.ht):
			t_idx = np.where(self.times == ht)[0][0]
			for irep in range(self.nreps[itpt]):
				for igen in range(self.exp_max_div+1):
					cell = <DTYPE_t>cells_gen[igen, t_idx]
					model.append(cell)
		return np.asfarray(model)

	# following function is a copy of 'cyton15' but with extra dead cell computation
	def extrapolate(self, model_times, params):
		# Stimulated cells parameters
		cdef DTYPE_t mDiv0 = params['mDiv0']
		cdef DTYPE_t sDiv0 = params['sDiv0']
		cdef DTYPE_t mDD = params['mDD']
		cdef DTYPE_t sDD = params['sDD']
		cdef DTYPE_t mDie = params['mDie']
		cdef DTYPE_t sDie = params['sDie']
		
		# Subsequent division time
		cdef DTYPE_t m = params['m']

		cdef unsigned int n = model_times.size

		# Compute pdf
		cdef np.ndarray[DTYPE_t, ndim=1] pdfDD = np.zeros(shape=n, dtype=DTYPE)
		pdfDD = self.compute_pdf(model_times, mDD, sDD)

		# Compute 1 - cdf
		cdef np.ndarray[DTYPE_t, ndim=1] sfDiv = np.zeros(shape=n, dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1] sfDie = np.zeros(shape=n, dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=1] sfDD = np.zeros(shape=n, dtype=DTYPE)
		sfDiv = self.compute_sf(model_times, mDiv0, sDiv0)
		sfDie = self.compute_sf(model_times, mDie, sDie)
		sfDD = self.compute_sf(model_times, mDD, sDD)

		# declare 2 arrays for divided cells & destiny cells
		cdef np.ndarray[DTYPE_t, ndim=2] nDIV = np.zeros(shape=(self.max_div+1, n), dtype=DTYPE)
		cdef np.ndarray[DTYPE_t, ndim=2] nDES = np.zeros(shape=(self.max_div+1, n), dtype=DTYPE)

		# store number of cells at all time per generation
		cdef np.ndarray[DTYPE_t, ndim=2] cells_gen = np.zeros(shape=(self.exp_max_div+1, n), dtype=DTYPE)

		# store total live cells
		cdef np.ndarray[DTYPE_t, ndim=1] total_live_cells = np.zeros(shape=n, dtype=DTYPE)

		# calculate gen = 0 cells
		cdef DTYPE_t x, y
		nDIV[0,:] = self.n0 * sfDie * sfDiv * sfDD
		nDES[0,:] = self.n0 * sfDie * np.cumsum([x * y for x, y in zip(pdfDD, sfDiv)]) * self.dt
		cells_gen[0,:] = nDIV[0,:] + nDES[0,:]  # cells in generation 0

		# calculate gen > 0 cells
		cdef DTYPE_t core
		cdef unsigned int igen
		for igen in range(1, self.max_div+1):
			core = <DTYPE_t>(2.**igen * self.n0)

			upp_cdfDiv = self.compute_cdf(model_times - <DTYPE_t>((igen - 1.)*m), mDiv0, sDiv0)
			low_cdfDiv = self.compute_cdf(model_times - <DTYPE_t>(igen*m), mDiv0, sDiv0)
			difference = upp_cdfDiv - low_cdfDiv

			nDIV[igen,:] = core * sfDie * sfDD * difference
			nDES[igen,:] = core * sfDie * np.cumsum([x * y for x, y in zip(pdfDD, difference)]) * self.dt

			if igen < self.exp_max_div:
				cells_gen[igen,:] = nDIV[igen,:] + nDES[igen,:]
			else:
				cells_gen[self.exp_max_div,:] += nDIV[igen,:] + nDES[igen,:]
		total_live_cells = np.sum(cells_gen, axis=0)  # sum over all generations per time point

		cdef unsigned int itpt
		cdef list cells_gen_at_ht = [[] for _ in range(len(self.ht))]
		cdef np.ndarray[DTYPE_t, ndim=1] total_live_cells_at_ht = np.zeros(shape=(len(self.ht)), dtype=DTYPE)
		for itpt, ht in enumerate(self.ht):
			t_idx = np.where(model_times == ht)[0][0]
			for igen in range(self.exp_max_div+1):
				cells_gen_at_ht[itpt].append(cells_gen[igen, t_idx])
			total_live_cells_at_ht[itpt] = total_live_cells[t_idx]

		res = {
			'ext': {  # Extrapolated cell numbers
				'total_live_cells': total_live_cells,
				'cells_gen': cells_gen,
				'nDIV': nDIV, 'nDES': nDES
			},
			'hts': {  # Collect cell numbers at harvested time points
				'total_live_cells': total_live_cells_at_ht,
				'cells_gen': cells_gen_at_ht
			}
		}
		
		return res