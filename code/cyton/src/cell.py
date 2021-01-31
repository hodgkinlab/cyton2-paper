"""
Cyton 2 Agent-based simulation
Last edit: 24-October-2020

This code is largely based on Gianfelice's code.
The simulation is semi-deterministic where,
	Times to first div, die and destiny are randomly drawn from lognormal distribution, respectively
	Except, we prescribe a constant subseqeunt division time equal for all generation > 0.

It's an identical code that I used for ABM simulation in "tracking" folder. But my focus is to recreate Marchingo et al. Scien 2014 Fig3 results, a linear sum of mean division number from different costimulation. Here, we write a separate function to extract mean division number given a collection of family trees.
"""
import tqdm
import numpy as np
import pandas as pd
import scipy.stats as sps
import multiprocessing as mp
rng = np.random.RandomState(seed=50855782)

# class cell:
# 	## Assume we can always observe the final fate of the cells
# 	def __init__(self, gen, t_start, t_end, ttfd, avgSubDiv, gDestiny, gDeath, flag_destiny=False):
# 		self.gen = gen                      # Generation
# 		self.born = t_start                 # Birth time
# 		self.life = None  					# Life time
# 		self.destiny = None					# Destiny time
# 		self.fate = None
# 		self.flag_destiny = flag_destiny    # Flag destiny

# 		## Determine cell's fate separately for gen=0 and gen>0
# 		if gen == 0:
# 			## Check if death occurs first
# 			if gDeath <= min(gDestiny, ttfd):
# 				self.destiny = np.nan  # destiny time would be (explicitly) censored
# 				self.fate = "died"
# 				self.life = gDeath
# 				self.left = None; self.right = None
# 			## Check if cell divides
# 			elif ttfd < min(gDestiny, gDeath):
# 				self.fate = "divided"
# 				self.life = ttfd
# 				self.left = cell(self.gen+1, self.life, t_end, ttfd, avgSubDiv, gDestiny, gDeath, flag_destiny)
# 				self.right = cell(self.gen+1, self.life, t_end, ttfd, avgSubDiv, gDestiny, gDeath, flag_destiny)
# 			## Check if cell reached destiny
# 			elif gDestiny < min(ttfd, gDeath):
# 				self.destiny = gDestiny
# 				self.flag_destiny = True
# 				## Check current cell cycle progress
# 				if ttfd < gDeath:
# 					cell_progress = gDestiny / ttfd
# 					## If cell progressed division cycle more then 1/3, then allow it to divide but flag next gen cells to be destiny
# 					if cell_progress >= 1/3:
# 						self.fate = "divided"
# 						self.life = ttfd
# 						self.left = cell(self.gen+1, self.life, t_end, ttfd, avgSubDiv, gDestiny, gDeath, flag_destiny)
# 						self.right = cell(self.gen+1, self.life, t_end, ttfd, avgSubDiv, gDestiny, gDeath, flag_destiny)
# 					else:
# 						self.fate = "died"
# 						self.life = gDeath
# 						self.left = None; self.right = None
# 				else:
# 					self.fate = "died"
# 					self.life = gDeath
# 					self.left = None; self.right = None
# 		else:
# 			## Check if cell was reached destiny and allowed divide once from previous generation
# 			if self.flag_destiny:
# 				self.destiny = gDestiny
# 				self.fate = "died"
# 				self.life = gDeath - self.born
# 				self.left = None; self.right = None
# 			else:
# 				if gDeath <= min(gDestiny, self.born + avgSubDiv):
# 					self.fate = "died"
# 					self.life = gDeath - self.born
# 					self.left = None; self.right = None
# 				elif self.born + avgSubDiv < min(gDestiny, gDeath):
# 					self.fate = "divided"
# 					self.life = avgSubDiv
# 					self.left = cell(self.gen+1, self.born+self.life, t_end, ttfd, avgSubDiv, gDestiny, gDeath, flag_destiny)
# 					self.right = cell(self.gen+1, self.born+self.life, t_end, ttfd, avgSubDiv, gDestiny, gDeath, flag_destiny)
# 				elif gDestiny < min(self.born + avgSubDiv, gDeath):
# 					self.destiny = gDestiny
# 					self.flag_destiny = True
# 					## Check current cell cycle progress
# 					if self.born + avgSubDiv < gDeath:
# 						cell_progress = (gDestiny - self.born) / avgSubDiv
# 						## If cell progressed division cycle more then 1/3, then allow it to divide but flag next gen cells to be destiny
# 						if cell_progress >= 1/3:
# 							self.fate = "divided"
# 							self.life = avgSubDiv
# 							self.left = cell(self.gen+1, self.born+self.life, t_end, ttfd, avgSubDiv, gDestiny, gDeath, flag_destiny)
# 							self.right = cell(self.gen+1, self.born+self.life, t_end, ttfd, avgSubDiv, gDestiny, gDeath, flag_destiny)
# 						else:
# 							self.fate = "died"
# 							self.life = gDeath - self.born
# 							self.left = None; self.right = None
# 					else:
# 						self.fate = "died"
# 						self.life = gDeath - self.born
# 						self.left = None; self.right = None
# 			# print(f'Cell gen={self.gen} -> born={self.born:.2f}, life={self.life:.2f}, flag={self.flag_destiny}')

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

	def run(self, name, pars, n_sims):
		self.n_sims = n_sims

		### Get timer parameters
		mT0, sT0 = pars['mDiv0'], pars['sDiv0']  # Time to first division
		mD, sD = pars['mDD'], pars['sDD'] 	     # Time to division destiny (Fit from Time to last division)
		mX, sX = pars['mDie'], pars['sDie']		 # Time to death
		m = pars['m']							 # Time to subsequent division

		pos = mp.current_process()._identity[0]-1  # For progress bar
		for _ in tqdm.trange(self.n_sims, desc=f"[{name}] Simulation", leave=False, position=2*pos+1):
			# Reset datarame for next iteration
			for col in self.df.columns:
				if col != 'time':
					self.df[col].values[:] = 0.0

			for _ in tqdm.trange(self.n0, desc=f"[{name}] Generate Trees", leave=False, position=2*pos+2):  # n_sim = number of clones!
				# TIMERS SETTING
				# T0 = sps.lognorm.rvs(sT0, scale=mT0, size=1, random_state=rng)[0]  # sample time to first division
				# D = sps.lognorm.rvs(sD, scale=mD, size=1, random_state=rng)[0]     # sample global destiny
				# X = sps.lognorm.rvs(sX, scale=mX, size=1, random_state=rng)[0]     # sample global death
				T0 = sps.norm.rvs(loc=mT0, scale=sT0, size=1, random_state=rng)[0]  # sample time to first division
				D = sps.norm.rvs(loc=mD, scale=sD, size=1, random_state=rng)[0]     # sample global destiny
				X = sps.norm.rvs(loc=mX, scale=sX, size=1, random_state=rng)[0]     # sample global death
				# print(f">> SETTING TIMERS\n   t0={T0:.2f}h, m={m:.2f}h, tdd={D:.2f}, tdie={X:.2f}h")

				tree = cell(gen=0, t0=self.times[0], tf=self.times[-1], ttfd=T0, avgSubDiv=m, gDestiny=D, gDeath=X)
				stat = self.tree_counts(tree, X).transpose()  # returns # cells and generation info
				cells = stat[0]
				gens = stat[1]/stat[0]

				for itpt, (nCell, igen) in enumerate(zip(cells, gens)):
					if not np.isnan(igen):
						if igen < self.max_gen:
							self.df.loc[itpt, f'gen{int(igen)}'] += nCell
						else:
							self.df.loc[itpt, f'gen{self.max_gen}'] += nCell
					else:
						break
			self.df['total'] = self.df.loc[:, [f'gen{int(igen)}' for igen in range(8+1)]].values.sum(axis=1)
			self.dfs.append(self.df.copy())