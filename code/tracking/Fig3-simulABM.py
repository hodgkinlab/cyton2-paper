"""
Cyton 2-like Agent-based simulation
Last edit: 10-February-2021

The simulation is semi-deterministic where,
	Times to first div, to die, to destiny and subsequent div time are randomly drawn from lognormal distribution, respectively
	The subseqeunt division time is set to equal for all generation > 0 (inherit).
Main purpose of the simulation is to validate censorship property we observed in the microscope data.
Run this script for generating Fig3 in the main article; 
	- Run for all the other datasets (i.e. Costim1 and Costim2; not included in the article)
"""
import os, time, datetime, copy, itertools
import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scipy as sp
import scipy.stats as sps
import multiprocessing as mp
mpl.use('Agg')
rng = np.random.RandomState(seed=63699138)

# GLOBAL PLOT SETTINGS
rc = {
	'font.size': 14, 'axes.titlesize': 14, 'axes.labelsize': 12,
	# 'xtick.labelsize': 14, 'ytick.labelsize': 14,
	'figure.figsize': (8, 6),
	# 'axes.grid': True, 'axes.grid.axis': 'x', 'axes.grid.axis': 'y',
	'axes.axisbelow': True, 'axes.titlepad': 0,
	'axes.spines.top': False, 'axes.spines.right': False,
	'axes.spines.left': True, 'axes.spines.bottom': True,
	'ytick.left': True, 'xtick.bottom': True,
	'lines.markersize': 7.5, 'lines.linewidth': 1.5,
	'errorbar.capsize': 2.5
}
sns.set(context='paper', style='white', rc=rc)

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

def _merge_dicts(a, b, complete):
	new_dict = copy.deepcopy(a)
	if complete:  # Merge into the same key
		for key, value in b.items():
			new_dict.setdefault(key, []).extend(value)
	else:  # create a new key *_[i] index if it coincides
		for key, value in b.items():
			if key in new_dict:
				# rename first key
				counter = itertools.count(1)
				while True:
					new_key = f'{key}_{next(counter)}'
					if new_key not in new_dict:
						new_dict[new_key] = new_dict.pop(key)
						break
				# create second key
				while True:
					new_key = f'{key}_{next(counter)}'
					if new_key not in new_dict:
						new_dict[new_key] = value
						break
			else:
				new_dict[key] = value
	return new_dict

def stat_tree_times(tree):
	stat = {
		'gen': [tree.gen],
		'born': [tree.born],
		'lifetime': [tree.life],
		'death': [tree.born + tree.life],
		'fate': [tree.fate]
	}

	if tree.left is None: 
		return stat
	else: 
		return _merge_dicts(a=stat, b=_merge_dicts(a=stat_tree_times(tree.left), b=stat_tree_times(tree.right), complete=True), complete=True)

# Count number of live cells
def stat_tree_counts(tree, times, gDeath):
	Z = []
	for t in times:
		if tree.fate == 'divided':
			if tree.born <= t < tree.born+tree.life:
				Z.append(1)
			else:
				Z.append(0)
		elif tree.fate == 'died':
			if tree.born <= t < tree.born+tree.life and tree.born <= t < gDeath:
				Z.append(1)
			else:
				Z.append(0)
	Z = np.array(Z)
	if tree.left is None: 
		return Z
	else: 
		return Z + stat_tree_counts(tree.left, times, gDeath) + stat_tree_counts(tree.right, times, gDeath)

def stat_tree_counts_destiny(tree, times, gDeath):
	D = []
	for t in times:
		if tree.flag_destiny:
			if tree.fate == 'divided':
				if tree.destiny <= t < tree.born+tree.life:
					D.append(1)
				else:
					D.append(0)
			elif tree.fate == 'died':
				if tree.destiny <= t < tree.born+tree.life and tree.born <= t < gDeath:
					D.append(1)
				else:
					D.append(0)
		else:
			D.append(0)
	D = np.array(D)
	if tree.left is None:
		return D
	else:
		return D + stat_tree_counts_destiny(tree.left, times, gDeath) + stat_tree_counts_destiny(tree.right, times, gDeath)

def plot_tree(tree, ax, tf, y, ystep):
	xLeft = tree.born
	xRight = xLeft + tree.life

	fate = tree.fate
	if fate == 'divided':
		color = "blue"
	elif fate == 'died':
		if xRight > tf:    # End of Simulation (but would observe death)
			color = 'gray'
		else:
			color = "red"

	ax.plot([xLeft, xRight], [y, y], color=color)

	if tree.left is None:
		return y
	else:
		yTop = y - ystep
		yBtm = y + ystep
		ax.plot([xRight, xRight], [yTop, yBtm], color=color)
		return plot_tree(tree.left, ax, tf, yTop, ystep/2), plot_tree(tree.right, ax, tf, yBtm, ystep/2)

# def run_simulation(name, df, par, t0, tf, n_sim):
def run_simulation(inputs):
	def diag(x, **kws):  # access diagonal plots in seaborn's PairGrid object
		ax = plt.gca()
		lab = kws['label']

		ax.set_rasterized(True)
		if lab == 'Obs. time':
			ax.annotate(f"$N_{{Obs.}}={len(x)}$", xy=(0.02, 0.88), xycoords=ax.transAxes, color=kws['color'])
		ax.set_xlim(left=0)

	def corrfunc(x, y, **kws):  # access off-diagonal plots
		ax = plt.gca()
		ax.set_rasterized(True)

		lab = kws['label']

		xy = np.array([x, y]).T
		n, _ = np.shape(xy)
		freqr = np.corrcoef(xy[:, 0], xy[:, 1])

		## Exact solution Jeffreys (numerical integration) Theory of Probability (1961), pp. 291 Eq.(9):
		## Or nicely presented in Wagenmakers et al. 2016 Appendix
		# bf10 = sp.integrate.quad(  # BF10
		# 	lambda rho: ((1. - rho ** 2.) ** ((n - 1.) / 2.))
		# 	/ ((1. - rho * freqr[0, 1]) ** ((n - 1.) - 0.5)),
		# 	-1., 1.
		# )[0] * 0.5

		## We need to use approximate form due to large n number. The integration becomes numerically unstable.
		bf10 = 1. / (((2. * (n - 1.) - 1.) / np.pi) ** 0.5 * (1. - freqr[0, 1] ** 2.) ** (0.5 * ((n - 1.) - 3.)))  # BF10

		if bf10 > 1:  # BF10: Favours the alternative hypothesis (rho != 0)
			if bf10 == 1.:
				string = r"BF$_{10}$" + f" = {bf10:.2f}"; color = "#000000"
			elif 1 < bf10 < 3:
				string = r"BF$_{10}$" + f" = {bf10:.2f}"; color = "#000000"
			elif 3 < bf10 < 10:
				string = r"BF$_{10}$" + f" = {bf10:.2f}"; color = "#400000"
			elif 10 < bf10 < 30:
				string = r"BF$_{10}$" + f" = {bf10:.2f}"; color = "#800000"
			elif 30 < bf10 < 100:
				string = r"BF$_{10}$" + f" = {bf10:.2f}"; color = "#BF0000"
			elif bf10 > 100:
				string = r"BF$_{10}$" + f" > 100"; color = "#FF0000"
			elif np.isnan(bf10):
				string = r"BF$_{10}$ NaN"; color = "#000000"
		else:  # BF01: Favours the null hypothesis (rho = 0)
			bf01 = 1/bf10
			if bf01 == 1.:
				string = r"BF$_{01}$" + f" = {bf01:.2f}"; color = "#000000"
			elif 1 < bf01 < 3:
				string = r"BF$_{01}$" + f" = {bf01:.2f}"; color = "#000000"
			elif 3 < bf01 < 10:
				string = r"BF$_{01}$" + f" = {bf01:.2f}"; color = "#000080"
			elif 10 < bf01 < 30:
				string = r"BF$_{01}$" + f" = {bf01:.2f}"; color = "#061D95"
			elif 30 < bf01 < 100:
				string = r"BF$_{01}$" + f" = {bf01:.2f}"; color = "#0D3AA9"
			elif bf01 > 100:
				string = r"BF$_{01}$" + f" > 100"; color = "#1974D2"
			elif np.isnan(bf01):
				string = r"BF$_{01}$ NaN"; color = "#000000"

		if lab == 'True time':
			ax.annotate("[True] " + string, xy=(.02, .88), xycoords=ax.transAxes, color=color, fontsize='medium')
		elif lab == 'Obs. time':
			ax.annotate("[Obs.] " + string, xy=(.02, .75), xycoords=ax.transAxes, color=color, fontsize='medium')

		ax.set_xlim(left=0)
		ax.set_ylim(bottom=0)

	name, df, par, t0, tf, n_sim = inputs

	pos = mp.current_process()._identity[0]-1  # For progress bar
	times = np.linspace(t0, tf+1, num=5000)

	mT0, sT0 = par['mT0'], par['sT0']
	mT, sT = par['mT'], par['sT']
	mD, sD = par['mD'], par['sD']
	mX, sX = par['mX'], par['sX']

	# function that generates n_simul trees that survive to the end
	# Store timers
	TIME_TO_FIRST_DIV = []
	TIME_TO_SUB_DIV = []
	TIME_TO_DESTINY = []
	TIME_TO_DIE = []
	LABEL = []

	## Location to save
	save_path = "./out/Fig3-Simulation/" + name.replace('/','_')
	if not os.path.exists(save_path):
		save_folder = os.mkdir(save_path)
	else:
		save_folder = save_path
	save_path_real = save_path + "/_clones"
	if not os.path.exists(save_path_real):
		save_folder2 = os.mkdir(save_path_real)
	else:
		save_folder2 = save_path_real

	random_clones = rng.randint(low=1, high=n_sim, size=1000)  # select 1000 random clones to inspect...
	for counter in tqdm.trange(n_sim, desc=f"[{name}] Generate Trees", leave=False, position=pos+1):
		# TIMERS SETTING
		T0 = sps.lognorm.rvs(sT0, scale=mT0, size=1, random_state=rng)[0]  # sample time to first division
		T = sps.lognorm.rvs(sT, scale=mT, size=1, random_state=rng)[0]  # sample subsequent division time and pass down to the progenies
		D = sps.lognorm.rvs(sD, scale=mD, size=1, random_state=rng)[0]  # sample global destiny
		X = sps.lognorm.rvs(sX, scale=mX, size=1, random_state=rng)[0]  # sample global death

		tree = cell(gen=0, t0=0, tf=tf, ttfd=T0, avgSubDiv=T, gDestiny=D, gDeath=X)
		stat = pd.DataFrame(stat_tree_times(tree))

		# collapsed the tree
		tdiv0 = stat[(stat['fate']=='divided') & (stat['gen']==0)].lifetime
		if len(tdiv0) > 0: tdiv0 = tdiv0.values[0]
		else: tdiv0 = np.nan
		tdiv = np.mean(stat[(stat['fate']=='divided') & (stat['gen']>0)].lifetime)
		tdie = np.mean(stat[stat['fate']=='died'].death)
		tld = np.mean(stat[stat['fate']=='died'].born)
		if tld == 0.0: tld = np.nan  # died before first division

		# Record true values
		TIME_TO_FIRST_DIV.append(T0)
		TIME_TO_SUB_DIV.append(T)
		TIME_TO_DESTINY.append(D)
		TIME_TO_DIE.append(X)
		LABEL.append("True time")

		TIME_TO_FIRST_DIV.append(tdiv0)
		TIME_TO_SUB_DIV.append(tdiv)
		TIME_TO_DESTINY.append(tld)
		TIME_TO_DIE.append(tdie)
		LABEL.append("Obs. time")

		# Get family members
		if counter in random_clones or X < min([T0, T, D]):
			Zg = stat_tree_counts(tree, times, gDeath=X)
			Zg_des = stat_tree_counts_destiny(tree, times, gDeath=X)

			first_destiny = next((i for i, x in enumerate(Zg_des) if x), None)
			max_cell = max(max(Zg), max(Zg_des))

			#########################################################################################################
			# 									      Plot total cells												#
			#########################################################################################################
			fig, ax = plt.subplots()
			ax.set_title(f"Clone #{counter}", x=0.01, ha='left', weight='bold', fontsize='x-large')
			ax.set_ylabel("Cell number")
			ax.set_xlabel("Time (hour)")

			if first_destiny is not None:
				ax.plot(times[:first_destiny+1], Zg[:first_destiny+1], 'blue', label='Dividing')
				ax.plot(times[first_destiny:], Zg_des[first_destiny:], '--', c='green', label='Destiny')
				ax.scatter(D, Zg_des[first_destiny], marker='*', c='green', ec='k', zorder=2.5)
				ax.annotate(f"$t_{{dd}}={D:.2f}h$", xy=(D, Zg_des[first_destiny]), xycoords="data", xytext=(0.01, 0.7), textcoords='axes fraction', 
							arrowprops=dict(facecolor='forestgreen', shrink=0), va='bottom', color='forestgreen', zorder=-1, fontsize='large')
			else:
				ax.plot(times, Zg, 'blue', label='Dividing')
				ax.scatter(D, 0, marker='*', c='green', ec='k', zorder=2.5)  # where destiny would have been
				ax.annotate(f"$t_{{dd}}={D:.2f}h$", xy=(D, 0), xycoords="data", xytext=(0.01, 0.7), textcoords='axes fraction', 
							arrowprops=dict(facecolor='forestgreen', shrink=0), va='bottom', color='forestgreen', zorder=-1, fontsize='large')
			if not np.isnan(tld):
				ax.scatter(tld, max_cell, c='green', ec='k', zorder=2.5)
				ax.annotate(f"$t_{{ld}}={tld:.2f}h$", xy=(tld, max_cell), xycoords="data", xytext=(0.01, 0.85), textcoords='axes fraction', 
							arrowprops=dict(facecolor='forestgreen', shrink=0), va='bottom', color='forestgreen', zorder=-1, fontsize='large')
			ax.scatter(T0, 1, c='blue', ec='k', zorder=2.5)
			ax.scatter(X, max_cell, marker='X', c='red', ec='k', zorder=2.5)

			# Annotate timers
			ax.text(s=f"$t_{{div}}^0={T0:.2f}h$", x=T0+3, y=1., ha='left', va='bottom', color='blue', fontsize='large')
			if X < tf:
				ax.text(s=f"$t_{{die}}={X:.2f}h$", x=X+3, y=max_cell, ha='left', va='bottom', color='red', fontsize='large')
			else:
				ax.text(s=f"$t_{{die}}={X:.2f}h$", x=tf-35, y=max_cell, ha='left', va='bottom', color='red', fontsize='large')

			# ax.spines['top'].set_visible(False)
			# ax.spines['right'].set_visible(False)
			ax.yaxis.set_major_locator(MaxNLocator(integer=True))
			ax.set_ylim(bottom=0)
			ax.set_xlim(left=0, right=tf)
			fig.legend(ncol=2, frameon=False, fontsize='large')
			fig.tight_layout(rect=(0, 0, 1, 1))
			fig.savefig(f"{save_path_real}/c{counter}_v1.pdf", dpi=300)
			plt.close()

			#########################################################################################################
			# 									      Plot family tree												#
			#########################################################################################################
			widths = [1]
			heights = [6, 1]
			gs_kw = dict(width_ratios=widths, height_ratios=heights)
			fig, axes = plt.subplots(nrows=2, sharex=True, gridspec_kw=gs_kw)
			
			plot_tree(tree, axes[0], tf, 0, 1)
			axes[0].set_title(f"Clone #{counter}", x=0.01, ha='left', weight='bold', fontsize='x-large')
			axes[0].axvline(T0, ls=':', c='blue', zorder=-1)
			axes[0].axvline(tld, ls=':', c='forestgreen', zorder=-1)
			axes[0].axvline(D, ls=':', c='forestgreen', zorder=-1)
			axes[0].axvline(tdie, ls=':', c='red', zorder=-1)
			
			axes[1].scatter(T0, 1, marker='o', ec='k', c='blue', zorder=10, label=f'$t_{{div}}^0 = {T0:.2f}$h')
			axes[1].axvline(T0, ls=':', c='blue', zorder=-1)
			axes[1].annotate(r"$\rightarrow$" + f"$m={T:.2f}$h", xy=(T0+1, 0.58), xycoords=('data', 'axes fraction'))
			axes[1].scatter(tld, 1, marker='o', ec='k', c='forestgreen', zorder=10, label=f'$t_{{ld}} = {tld:.2f}$h')
			axes[1].axvline(tld, ls=':', c='forestgreen', zorder=-1)
			axes[1].scatter(D, 1, marker='*', ec='k', c='forestgreen', zorder=10, label=f'$t_{{dd}} = {D:.2f}$h')
			axes[1].axvline(D, ls=':', c='forestgreen', zorder=-1)
			if tdie <= X:
				axes[1].plot([0, tdie], [1, 1], 'k', zorder=-1)
				axes[1].scatter(tdie, 1, marker='X', ec='k', c='red', zorder=10, label=f'$t_{{die}} = {tdie:.2f}$h')
				axes[1].axvline(tdie, ls=':', c='red', zorder=-1)
			else:
				axes[1].plot([0, tf], [1, 1], 'k', zorder=-1)
				axes[1].scatter(tdie, 1, marker='X', ec='k', c='red', zorder=10, label=f'$t_{{die}} = {X:.2f}$h')

			axes[0].xaxis.set_visible(False)
			axes[0].spines['bottom'].set_visible(False)
			axes[1].set_xlim(left=t0, right=tf)
			axes[1].set_xlabel("Time (hour)")
			for ax in axes:
				ax.yaxis.set_visible(False)
				ax.spines['left'].set_visible(False)
			fig.legend(ncol=1, frameon=False, fontsize='large', handletextpad=0.01, columnspacing=0)
			fig.tight_layout(rect=(0, 0, 1, 1))
			fig.subplots_adjust(hspace=0, wspace=0)

			fig.savefig(f"{save_path_real}/c{counter}_v2.pdf", dpi=300)
			plt.close()

	df_sim = pd.DataFrame({
		"tdiv0": TIME_TO_FIRST_DIV,
		"tdiv": TIME_TO_SUB_DIV,
		"tld": TIME_TO_DESTINY,
		"tdie": TIME_TO_DIE,
		"Label": LABEL
	})

	cols = ['#e6194B', '#003f5c']
	bin_list = np.arange(start=0, stop=np.nanmax(df_sim.max(numeric_only=True)), step=1)
	g = sns.pairplot(df_sim, hue="Label", markers=["o", "o"], palette=sns.color_palette(cols), height=1.5, corner=True, diag_kind='hist', 
						diag_kws=dict(bins=bin_list, density=True, ec='k', lw=1, alpha=0.3), 
						plot_kws=dict(s=22, ec='none', linewidth=1, alpha=0.2, rasterized=True), 
						grid_kws=dict(diag_sharey=False))
	g._legend.remove()
	g.map_diag(diag)
	g.map_lower(corrfunc)

	data_color = '#f58231'
	for i, j in zip(*np.tril_indices_from(g.axes, 0)):
		_, xmax = g.axes[i,j].get_xlim()
		_, ymax = g.axes[i,j].get_ylim()
		if xmax < ymax: smaller = xmax
		else: smaller = ymax
		straight_x = np.linspace(0, smaller, num=1000)
		straight_y = np.linspace(0, smaller, num=1000)

		if i == 1 and j == 0:
			g.axes[i,j].spines['top'].set_visible(True)
			g.axes[i,j].spines['right'].set_visible(True)
			g.axes[i,j].scatter(df['tdiv0'], df['tdiv'], c=data_color, marker='o', ec='k', lw=0.5, s=18, label='Data', rasterized=True)
			g.axes[i,j].set_ylabel(r"Avg. sub div time ($M$)")
		elif i == 2 and j == 0:
			g.axes[i,j].spines['top'].set_visible(True)
			g.axes[i,j].spines['right'].set_visible(True)
			g.axes[i,j].plot(straight_x, straight_y, '--k', lw=1, zorder=-1)
			g.axes[i,j].scatter(df['tdiv0'], df['tld'], c=data_color, marker='o', ec='k', lw=0.5, s=18, rasterized=True)
			g.axes[i,j].set_ylabel(r"Time to last div ($T_{ld}$)")
		elif i == 3 and j == 0:
			g.axes[i,j].spines['top'].set_visible(True)
			g.axes[i,j].spines['right'].set_visible(True)
			g.axes[i,j].plot(straight_x, straight_y, '--k', lw=1, zorder=-1)
			g.axes[i,j].scatter(df['tdiv0'], df['tdie'], c=data_color, marker='o', ec='k', lw=0.5, s=18, rasterized=True)
			g.axes[i,j].set_xlabel(r"Time to first div ($T_{div}^0$)")
			g.axes[i,j].set_ylabel(r"Time to death ($T_{die}$)")
			g.axes[i,j].set_xlim(left=0, right=max(df_sim['tdiv0']))
		elif i == 2 and j == 1:
			g.axes[i,j].spines['top'].set_visible(True)
			g.axes[i,j].spines['right'].set_visible(True)
			g.axes[i,j].scatter(df['tdiv'], df['tld'], c=data_color, marker='o', ec='k', lw=0.5, s=18, rasterized=True)
		elif i == 3 and j == 1:
			g.axes[i,j].spines['top'].set_visible(True)
			g.axes[i,j].spines['right'].set_visible(True)
			g.axes[i,j].scatter(df['tdiv'], df['tdie'], c=data_color, marker='o', ec='k', lw=0.5, s=18, rasterized=True)
			g.axes[i,j].set_xlabel(r"Avg. sub div time ($M$)")
			g.axes[i,j].set_xlim(left=0, right=max(df_sim['tdiv']))
		elif i == 3 and j == 2:
			g.axes[i,j].spines['top'].set_visible(True)
			g.axes[i,j].spines['right'].set_visible(True)
			g.axes[i,j].plot(straight_x, straight_y, '--k', lw=1, zorder=-1)
			g.axes[i,j].scatter(df['tld'], df['tdie'], c=data_color, marker='o', ec='k', lw=0.5, s=18, rasterized=True)
			g.axes[i,j].set_xlabel(r"Time to last div ($T_{ld}$)")
			g.axes[i,j].set_xlim(left=0, right=max(df_sim['tld']))
	g.axes[3,3].set_xlabel(r"Time to death ($T_{die}$)")
	handles, labels = g.axes[1,0].get_legend_handles_labels()
	leg = g.fig.legend(handles=handles, labels=labels, loc='center right', bbox_to_anchor=(0.88, 0.4), fontsize=12, ncol=1)  # with data (different anchor value)
	for lh in leg.legendHandles:
		lh.set_alpha(1)
		lh._sizes = [40]
	g.fig.set_size_inches(12, 9)
	g.fig.tight_layout(rect=(0, 0, 1, 1))
	g.fig.subplots_adjust(hspace=0.07, wspace=0.07)


	## Add interna plot for simulation setup
	left, bottom, width, height = [0.6, 0.6, 0.38, 0.3]
	in_ax = g.fig.add_axes([left, bottom, width, height])

	tt = np.linspace(t0, tf, num=10000)
	in_ax.set_title(f"Cyton distribution (Fitted parameters)")
	in_ax.set_ylabel("Prob. density")
	in_ax.set_xlabel("Time (hour)")
	in_ax.fill_between(tt, sps.lognorm.pdf(tt, sT0, scale=mT0), color='blue', label=f'$T_{{div}}^0 \sim LN({mT0}, {sT0})$', alpha=0.3)
	in_ax.fill_between(tt, sps.lognorm.pdf(tt, sT, scale=mT), color='orange', label=f"$M \sim LN({mT}, {sT})$", alpha=0.3)
	in_ax.fill_between(tt, sps.lognorm.pdf(tt, sD, scale=mD), color='green', label=f'$T_{{ld}} = T_{{dd}} \sim LN({mD}, {sD})$', alpha=0.3)
	in_ax.fill_between(tt, sps.lognorm.pdf(tt, sX, scale=mX), color='red', label=f'$T_{{die}} \sim LN({mX}, {sX})$', alpha=0.3)
	in_ax.spines['top'].set_visible(True)
	in_ax.spines['right'].set_visible(True)
	in_ax.set_ylim(bottom=0)
	in_ax.set_xlim(left=t0, right=tf)
	in_ax.legend(loc=1, frameon=False, fontsize=12)

	g.fig.savefig(f"{save_path}/summary_corr.pdf", dpi=300)


VAR_NAMES = iter(['tdiv0', 'tdiv', 'tld', 'tdie'])
if __name__ == "__main__":
	start = time.time()
	loc_data = './data/_processed/collapsed_times'
	path_data = [os.path.join(loc_data, file) for file in os.listdir(loc_data) if not file.startswith('.')]

	data = {
		'b_cpg/cpg3': path_data[path_data.index(loc_data + '/' + 'b_cpg_cpg3_0.0.csv')],
		'b_cpg/cpg4': path_data[path_data.index(loc_data + '/' + 'b_cpg_cpg4_0.0.csv')],
		't_il2/1U_aggre': path_data[path_data.index(loc_data + '/' + 'aggre_1U_1.0.csv')], 
		't_il2/3U_aggre': path_data[path_data.index(loc_data + '/' + 'aggre_3U_2.0.csv')],
		't_il2/10U_aggre': path_data[path_data.index(loc_data + '/' + 'aggre_10U_3.0.csv')],
		't_misc_20140211_1.0': path_data[path_data.index(loc_data + '/' + 't_misc_20140211_1.0.csv')], 
		't_misc_20140211_2.0': path_data[path_data.index(loc_data + '/' + 't_misc_20140211_2.0.csv')], # problematic one, I cannot get estimate on sub div time as there's no data...
		't_misc_20140211_3.0': path_data[path_data.index(loc_data + '/' + 't_misc_20140211_3.0.csv')],  # problematic one, no data for sub div time
		't_misc_20140211_4.0': path_data[path_data.index(loc_data + '/' + 't_misc_20140211_4.0.csv')], 
		't_misc_20140325_1.0': path_data[path_data.index(loc_data + '/' + 't_misc_20140325_1.0.csv')], 
		't_misc_20140325_2.0': path_data[path_data.index(loc_data + '/' + 't_misc_20140325_2.0.csv')], 
		't_misc_20140325_3.0': path_data[path_data.index(loc_data + '/' + 't_misc_20140325_3.0.csv')],
		't_misc_20140325_4.0': path_data[path_data.index(loc_data + '/' + 't_misc_20140325_4.0.csv')]
	}

	pars = {  ### RECREATE FILMING DATA RESULTS (LOGNORMAL)
		'b_cpg/cpg3': {'mT0': 38.4, 'sT0': 0.13, 'mT': 10.90, 'sT': 0.23, 'mD': 53.79, 'sD': 0.21, 'mX': 86.66, 'sX': 0.18},
		'b_cpg/cpg4': {'mT0': 41.51, 'sT0': 0.14, 'mT': 12.12, 'sT': 0.24, 'mD': 58.73, 'sD': 0.22, 'mX': 86.92, 'sX': 0.22},
		't_il2/1U_aggre': {'mT0': 35.73, 'sT0': 0.12, 'mT': 19.11, 'sT': 0.39, 'mD': 37.52, 'sD': 0.16, 'mX': 44.75, 'sX': 0.23},
		't_il2/3U_aggre': {'mT0': 41.26, 'sT0': 0.15, 'mT': 17.64, 'sT': 0.5, 'mD': 46.53, 'sD': 0.21, 'mX': 65.1, 'sX': 0.28},
		't_il2/10U_aggre': {'mT0': 41.97, 'sT0': 0.18, 'mT': 16.84, 'sT': 0.19, 'mD': 47.56, 'sD': 0.21, 'mX': 63.88, 'sX': 0.27},
		't_misc_20140211_1.0': {'mT0': 34.12, 'sT0': 0.11, 'mT': 14.03, 'sT': 0.25, 'mD': 39.96, 'sD': 0.23, 'mX': 48.18, 'sX': 0.21},
		't_misc_20140211_2.0': {'mT0': 33.52, 'sT0': 0.12, 'mT': 10.0, 'sT': 0.2, 'mD': 34.19, 'sD': 0.13, 'mX': 39.53, 'sX': 0.14}, # problematic one, I cannot get estimate on sub div time as there's no data...
		't_misc_20140211_3.0': {'mT0': 32.85, 'sT0': 0.1, 'mT': 10.0, 'sT': 0.2, 'mD': 32.85, 'sD': 0.1, 'mX': 36.74, 'sX': 0.13}, # problematic one, no data for sub div time
		't_misc_20140211_4.0': {'mT0': 35.77, 'sT0': 0.1, 'mT': 16.04, 'sT': 0.2, 'mD': 37.3, 'sD': 0.14, 'mX': 43.82, 'sX': 0.19},
		't_misc_20140325_1.0': {'mT0': 34.09, 'sT0': 0.10, 'mT': 10.24, 'sT': 0.2, 'mD': 42.52, 'sD': 0.2, 'mX': 48.13, 'sX': 0.22},
		't_misc_20140325_2.0': {'mT0': 39.25, 'sT0': 0.14, 'mT': 12.82, 'sT': 0.38, 'mD': 48.38, 'sD': 0.18, 'mX': 54.27, 'sX': 0.13},
		't_misc_20140325_3.0': {'mT0': 34.5, 'sT0': 0.07, 'mT': 10.28, 'sT': 0.17, 'mD': 41.35, 'sD': 0.15, 'mX': 46.9, 'sX': 0.17},
		't_misc_20140325_4.0': {'mT0': 34.12, 'sT0': 0.09, 'mT': 11.5, 'sT': 0.23, 'mD': 39.17, 'sD': 0.19, 'mX': 43.6, 'sX': 0.18}, 
	}

	# change your simulation time here
	t0, tf = 0, 140
	
	inputs = []
	for (key, datum), (_, par) in zip(data.items(), pars.items()):
		df = pd.read_csv(datum)
		df['Label'] = "Data"
		inputs.append((key, df, par, t0, tf, 1000000))

	tqdm.tqdm.set_lock(mp.RLock())  # for managing output contention
	p = mp.Pool(initializer=tqdm.tqdm.set_lock, initargs=(tqdm.tqdm.get_lock(),))
	with tqdm.tqdm(total=len(inputs), desc="Total", position=0) as pbar:
		for i, _ in enumerate(p.imap_unordered(run_simulation, inputs)):
			pbar.update()

	end = time.time()
	hours, rem = divmod(end-start, 3600)
	minutes, seconds = divmod(rem, 60)
	now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
	print(f"> DONE SIMULATION ! {now}")
	print("> Elapsed Time = {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
	