"""
Last edit: 16-May-2021

Reduced Cyton2 Agent-based simulation (source code)
The simulation rules:
	1. Times to first div, to die, to destiny and subsequent div time are randomly drawn from NORMAL distribution, respectively (+inherit)
	2. A constant sub div time for all generation > 0
NB: It's slightly different configuration than that of "Fig3-simulABM.py" to match with the reduced Cyton2 model
"""
import tqdm
import numpy as np
import pandas as pd
import scipy.stats as sps
import multiprocessing as mp
rng = np.random.RandomState(seed=50855782)

class cell:
	def __init__(self, gen, t0, tf, ttfd, avgSubDiv, gDestiny, gDeath, flag_destiny=False):
		self.gen = gen                      # Generation
		self.born = t0                 		# Birth time
		self.life = None  					# Life time
		self.destiny = None					# Destiny time
		self.fate = None
		self.flag_destiny = flag_destiny    # Flag destiny

		## Determine cell's fate separately for gen=0 and gen>0
		if gen == 0:
			## Check if death occurs first
			if gDeath <= min(gDestiny, ttfd):
				self.fate = "died"
				self.destiny = np.nan
				self.life = gDeath
				self.left = None; self.right = None
			## Check if cell divides
			elif ttfd < min(gDestiny, gDeath):
				self.fate = "divided"
				self.life = ttfd
				self.left = cell(self.gen+1, self.life, tf, ttfd, avgSubDiv, gDestiny, gDeath, flag_destiny)
				self.right = cell(self.gen+1, self.life, tf, ttfd, avgSubDiv, gDestiny, gDeath, flag_destiny)
			## Check if cell reached destiny
			elif gDestiny < min(ttfd, gDeath):
				self.destiny = gDestiny
				self.flag_destiny = True

				self.fate = "died"
				self.life = gDeath - self.born
				self.left = None; self.right = None
		else:
			## Check if cell was reached destiny and allowed divide once from previous generation
			if self.flag_destiny:
				self.fate = "died"
				self.life = gDeath - self.born
				self.left = None; self.right = None
			else:
				if gDeath <= min(gDestiny, self.born + avgSubDiv):
					self.fate = "died"
					self.life = gDeath - self.born
					self.left = None; self.right = None
				elif self.born + avgSubDiv < min(gDestiny, gDeath):
					self.fate = "divided"
					self.life = avgSubDiv
					self.left = cell(self.gen+1, self.born+self.life, tf, ttfd, avgSubDiv, gDestiny, gDeath, flag_destiny)
					self.right = cell(self.gen+1, self.born+self.life, tf, ttfd, avgSubDiv, gDestiny, gDeath, flag_destiny)
				elif gDestiny < min(self.born + avgSubDiv, gDeath):
					self.destiny = gDestiny
					self.flag_destiny = True

					self.fate = "died"
					self.life = gDeath - self.born
					self.left = None; self.right = None
			# print(f'Cell gen={self.gen} -> born={self.born:.2f}, life={self.life:.2f}, flag={self.flag_destiny}')

class ABM(cell):
	def __init__(self, rgs, t0, tf, dt, max_gen, n0):
		self.times = np.linspace(t0, tf, num=int(tf/dt)+1)
		self.max_gen = max_gen
		self.list_gens = np.array([f'gen{igen}' for igen in range(max_gen+1)])
		self.n0 = n0

		self.df = {f'gen{igen}': np.zeros(self.times.size) for igen in range(max_gen+1)}
		self.df['total'] = np.zeros(self.times.size)
		self.df['time'] = self.times
		self.df = pd.DataFrame(self.df)
		
		self.n_sims = 0
		self.rgs = rgs
		self.dfs = []  # store list of dfs

	# Count number of live cells
	def tree_counts(self, tree, gDeath):
		Z = []
		for t in self.times:
			if tree.born <= t < tree.born + tree.life:
				Z.append([1, tree.gen])
			else:
				Z.append([0, 0])
		Z = np.array(Z)
		if tree.left is None: 
			return Z
		else: 
			return Z + self.tree_counts(tree.left, gDeath) + self.tree_counts(tree.right, gDeath)
	
	# Calculate total cohort number
	def total_cohort(self, hts):
		self.total_cohort_hts = {'avg': [], 'low': [], 'upp': []}
		self.total_cohort_times = {'avg': [], 'low': [], 'upp': []}

		store1, store2 = [], []
		for df in self.dfs:
			tmp_hts, tmp_times = [], []
			for t in self.times:
				total_cohort = np.sum(df.loc[df['time']==t, self.list_gens].to_numpy().ravel() \
								* np.array([2**(-float(igen)) for igen in range(self.max_gen+1)]))
				tmp_times.append(total_cohort)
				if t in hts:
					tmp_hts.append(total_cohort)
			store1.append(tmp_hts)
			store2.append(tmp_times)

		alpha = (1. - self.rgs/100.)/2
		self.total_cohort_hts['avg'] = np.mean(np.array(store1), axis=0)
		self.total_cohort_hts['low'], self.total_cohort_hts['upp'] = np.quantile(np.array(store1), [alpha, alpha + self.rgs/100], interpolation='nearest', axis=0)
		self.total_cohort_times['avg'] = np.mean(np.array(store2), axis=0)
		self.total_cohort_times['low'], self.total_cohort_times['upp'] = np.quantile(np.array(store2), [alpha, alpha + self.rgs/100], interpolation='nearest', axis=0)

	# Calculate mean division number
	def mdn(self, hts):
		self.mdn_hts = {'avg': [], 'low': [], 'upp': []}
		self.mdn_times = {'avg': [], 'low': [], 'upp': []}

		store1, store2 = [], []
		for df in self.dfs:
			tmp_hts, tmp_times = [], []
			for t in self.times:
				cohort = df.loc[df['time']==t, self.list_gens].to_numpy().ravel() \
							* np.array([2**(-float(igen)) for igen in range(self.max_gen+1)])
				total_cohort = np.sum(cohort)
				mean_div_no = np.sum(cohort * np.array([float(igen) for igen in range(self.max_gen+1)])) / total_cohort
				tmp_times.append(mean_div_no)
				if t in hts:
					tmp_hts.append(mean_div_no)
			store1.append(tmp_hts)
			store2.append(tmp_times)

		alpha = (1. - self.rgs/100.)/2
		self.mdn_hts['avg'] = np.mean(np.array(store1), axis=0)
		self.mdn_hts['low'], self.mdn_hts['upp'] = np.quantile(np.array(store1), [alpha, alpha + self.rgs/100], interpolation='nearest', axis=0)
		self.mdn_times['avg'] = np.mean(np.array(store2), axis=0)
		self.mdn_times['low'], self.mdn_times['upp'] = np.quantile(np.array(store2), [alpha, alpha + self.rgs/100], interpolation='nearest', axis=0)

	def run(self, pos, name, pars, n_sims):
		self.n_sims = n_sims

		### Get timer parameters
		mT0, sT0 = pars['mDiv0'], pars['sDiv0']  # Time to first division
		mD, sD = pars['mDD'], pars['sDD'] 	     # Time to division destiny (Fit from Time to last division)
		mX, sX = pars['mDie'], pars['sDie']		 # Time to death
		m = pars['m']							 # Time to subsequent division
		for _ in tqdm.trange(self.n_sims, desc=f"[{name}] Simulation", leave=False, position=2*pos+1):
			# Reset datarame for next iteration
			for col in self.df.columns:
				if col != 'time': self.df[col].values[:] = 0.0

			for _ in tqdm.trange(self.n0, desc=f"[{name}] Generate Trees", leave=False, position=2*pos+2):  # n_sim = number of clones!
				# TIMERS SETTING
				T0 = sps.norm.rvs(loc=mT0, scale=sT0, random_state=rng)  # sample time to first division
				D = sps.norm.rvs(loc=mD, scale=sD, random_state=rng)     # sample global destiny
				X = sps.norm.rvs(loc=mX, scale=sX, random_state=rng)     # sample global death

				tree = cell(gen=0, t0=self.times[0], tf=self.times[-1], ttfd=T0, avgSubDiv=m, gDestiny=D, gDeath=X)
				stat = self.tree_counts(tree, X).transpose()  # returns # cells and generation info
				cells = stat[0]
				gens = stat[1]/stat[0]

				for itpt, (nCell, igen) in enumerate(zip(cells, gens)):
					if not np.isnan(igen):
						if igen < self.max_gen: self.df.loc[itpt, f'gen{int(igen)}'] += nCell
						else: self.df.loc[itpt, f'gen{self.max_gen}'] += nCell
					else: break
			self.df['total'] = self.df.loc[:, [f'gen{int(igen)}' for igen in range(8+1)]].values.sum(axis=1)
			self.dfs.append(self.df.copy())