"""
Last edit: 16-May-2021

Cyton2 Agent-based simulation
The simulation rules:
	1. Times to first div, to die, to destiny and sub div time are randomly drawn from LOG-NORMAL distribution, respectively (+inherit)
	2. The sub div times (per gen) are also randomly drawn from a LOG-NORMAL distribution, but cells of the same gen share the div time
Main purpose of the simulation is to validate censorship property we observed in the microscope data
[Output] Fig3 in the main article; 
	+ B-exp2, T-exp1 and T-exp2 (not included in the article)
"""
import sys, os, time, datetime, copy, itertools
import tqdm
import mpmath
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl; mpl.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scipy as sp
import scipy.stats as sps
import multiprocessing as mp
mpmath.dps = 100000000
rng = np.random.RandomState(seed=63699138)

# Check library versions
try:
	assert(sns.__version__=='0.10.1')
except AssertionError as ae:
	print("[VersionError] Please check if the following version of seaborn library is installed:")
	print("seaborn==0.10.1")
	sys.exit()

# GLOBAL PLOT SETTINGS
rc = {
	'figure.figsize': (8, 6),
	'font.size': 16, 
	'axes.titlesize': 16, 'axes.labelsize': 16,
	'xtick.labelsize': 16, 'ytick.labelsize': 16,
	'legend.fontsize': 16, 'legend.title_fontsize': None,
	# 'axes.grid': True, 'axes.grid.axis': 'x', 'axes.grid.axis': 'y',
	'axes.axisbelow': True, 'axes.titlepad': 0,
	'axes.spines.top': False, 'axes.spines.right': False,
	'axes.spines.left': True, 'axes.spines.bottom': True,
	'ytick.left': True, 'xtick.bottom': True,
	'lines.markersize': 7.5, 'lines.linewidth': 1,
	'errorbar.capsize': 2.5
}
sns.set(context='paper', style='white', rc=rc)

class cell:
	def __init__(self, gen, t0, tf, ttfd, avgSubDiv, gDestiny, gDeath, true_avgSubDiv, flag_destiny=False):
		self.gen = gen                      # Generation
		self.born = t0                 		# Birth time
		self.life = None  					# Life time
		self.destiny = None					# Destiny time
		self.fate = None					# Fate of cell (died or divided)
		self.flag_destiny = flag_destiny    # Flag destiny
		# self.avgSubDiv = avgSubDiv		# Cyton2.1 & 2.2: Either a constant or one random time, and inherit
		self.avgSubDiv = avgSubDiv[gen]		# Cyton2.3: Random times per gen

		## Determine cell's fate separately for gen=0 and gen>0
		if gen == 0:
			if gDeath <= min(gDestiny, ttfd):  # Check if death occurs first
				self.fate = "died"
				self.destiny = np.nan
				self.life = gDeath
				self.left = None; self.right = None
			elif ttfd < min(gDestiny, gDeath):  # Check if divides
				self.fate = "divided"
				self.life = ttfd
				self.left = cell(self.gen+1, self.life, tf, ttfd, avgSubDiv, gDestiny, gDeath, true_avgSubDiv, flag_destiny)
				self.right = cell(self.gen+1, self.life, tf, ttfd, avgSubDiv, gDestiny, gDeath, true_avgSubDiv, flag_destiny)
			elif gDestiny < min(ttfd, gDeath):  # Check if reached destiny
				self.destiny = gDestiny
				self.flag_destiny = True
				self.fate = "died"
				self.life = gDeath - self.born
				self.left = None; self.right = None
		else:
			if self.flag_destiny:  # prevents further division
				self.fate = "died"
				self.life = gDeath - self.born
				self.left = None; self.right = None
			else:
				true_avgSubDiv.append(self.avgSubDiv)
				if gDeath <= min(gDestiny, self.born + self.avgSubDiv):
					self.fate = "died"
					self.life = gDeath - self.born
					self.left = None; self.right = None
				elif self.born + self.avgSubDiv < min(gDestiny, gDeath):
					self.fate = "divided"
					self.life = self.avgSubDiv
					self.left = cell(self.gen+1, self.born+self.life, tf, ttfd, avgSubDiv, gDestiny, gDeath, true_avgSubDiv, flag_destiny)
					self.right = cell(self.gen+1, self.born+self.life, tf, ttfd, avgSubDiv, gDestiny, gDeath, true_avgSubDiv, flag_destiny)
				elif gDestiny < min(self.born + self.avgSubDiv, gDeath):
					self.destiny = gDestiny
					self.flag_destiny = True
					self.fate = "died"
					self.life = gDeath - self.born
					self.left = None; self.right = None

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

	ax.plot([xLeft, xRight], [y, y], lw=2, color=color)

	if tree.left is None:
		return y
	else:
		yTop = y - ystep
		yBtm = y + ystep
		ax.plot([xRight, xRight], [yTop, yBtm], lw=2, color=color)
		return plot_tree(tree.left, ax, tf, yTop, ystep/2), plot_tree(tree.right, ax, tf, yBtm, ystep/2)

# def run_simulation(name, df, par, t0, tf, n_sim):
def run_simulation(inputs):
	def diag(x, **kws):  # access diagonal plots in seaborn's PairGrid object
		ax = plt.gca()
		ax.set_rasterized(True)
		lab = kws['label']
		if lab == 'Obs. time':
			ax.annotate(f"$N_{{Obs.}}={len(x)}$", xy=(0.02, 0.88), xycoords=ax.transAxes, color=kws['color'], fontsize=16)
		ax.set_xlim(left=0)

	def corrfunc(x, y, **kws):  # access off-diagonal plots
		ax = plt.gca()
		ax.set_rasterized(True)
		lab = kws['label']
		xy = np.array([x, y]).T
		# n, _ = np.shape(xy)
		# sample_r = np.corrcoef(xy[:,0], xy[:,1])
		
		## Pick random samples of clones to avoid numerical issue with BF calculation
		total_n, _ = np.shape(xy)
		idx = rng.randint(total_n, size=10000)
		xy = xy[idx,:]
		n, _ = np.shape(xy)
		sample_r = np.corrcoef(xy[:,0], xy[:,1])

		## CALCULATE BAYESIAN FACTOR
		## https://github.com/pymc-devs/resources/blob/master/BCM/CaseStudies/ExtrasensoryPerception.ipynb
		## Approximate Jeffreys (1961), pp. 289-292 (use this to avoid numerical issue):
		# bf10 = 1. / (((2. * (n - 1.) - 1.) / np.pi) ** 0.5 * (1. - sample_r[0,1] ** 2.) ** (0.5 * ((n - 1.) - 3.)))
		# bf10 = float(1. / (mpmath.power((2. * (n - 1.) - 1.) / np.pi, 0.5) * mpmath.power(1. - sample_r[0,1] ** 2., 0.5 * ((n - 1.) - 3.))))

		## Exact solution Jeffreys (numerical integration) Theory of Probability (1961), pp. 291 Eq.(9):
		## Or nicely presented in Wagenmakers et al. 2016 Appendix
		# f_int = lambda rho: ((1. - rho ** 2.) ** ((n - 1.) / 2.)) / ((1. - rho * sample_r[0, 1]) ** (n - 3./2.)) # for large n, scipy quad becomes unstable
		# bf10 = 0.5 * sp.integrate.quad(f_int, -1, 1)[0]  
		f_int = lambda rho: mpmath.power(1. - rho**2., (n - 1.) / 2.) / mpmath.power(1. - rho * sample_r[0, 1], n - 3./2.) # higher precision
		bf10 = float(0.5 * mpmath.quad(f_int, [-1., 1.], error=False))

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
			# ax.annotate(r"[True] $r$" + f" = {freqr[0,1]:.4f}", xy=(0.02, 0.18), xycoords=ax.transAxes, fontsize='medium')
		elif lab == 'Obs. time':
			ax.annotate("[Obs.] " + string, xy=(.02, .75), xycoords=ax.transAxes, color=color, fontsize='medium')
			# ax.annotate(r"[Obs.] $r$" + f" = {freqr[0,1]:.4f}", xy=(0.02, 0.05), xycoords=ax.transAxes, fontsize='medium')

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
	if not os.path.exists(save_path): os.mkdir(save_path)
	else: pass

	save_path_real = save_path + "/clones"
	if not os.path.exists(save_path_real): os.mkdir(save_path_real)
	else: pass

	random_clones = rng.randint(low=1, high=n_sim, size=1000)  # select 100 random clones to plot trees
	for counter in tqdm.trange(n_sim, desc=f"[{name}] Generate Trees", leave=False, position=pos+1):
		# TIMERS SETTING
		T0 = sps.lognorm.rvs(sT0, scale=mT0, random_state=rng)  	   # sample time to first division
		# T = mT		   											   # (Cyton2.1) constant sub. div. time for all family & gens
		# T = sps.lognorm.rvs(sT, scale=mT, random_state=rng)		   # (Cyton2.2) sample one sub. div. time and pass down to the daughter cells
		T = sps.lognorm.rvs(sT, scale=mT, size=100, random_state=rng)  # (Cyton2.3) sample sub div times for each gen>0. Cells of the same gen share the time
		D = sps.lognorm.rvs(sD, scale=mD, random_state=rng)			   # sample global destiny
		X = sps.lognorm.rvs(sX, scale=mX, random_state=rng)			   # sample global death

		true_T = []
		tree = cell(gen=0, t0=0, tf=tf, ttfd=T0, avgSubDiv=T, gDestiny=D, gDeath=X, true_avgSubDiv=true_T)
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
		if np.isnan(np.mean(true_T)): T = T[1]  # NaN => cell reached destiny/death before first division. Record what it would be for gen 1 -> gen2
		else: T = np.mean(true_T)  # this includes censored values (comment this if-else statement for Cyton2.1 and 2.2)
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
		if counter in random_clones:
			#########################################################################################################
			# 									      Plot total cells												#
			#########################################################################################################
			# Zg = stat_tree_counts(tree, times, gDeath=X)
			# Zg_des = stat_tree_counts_destiny(tree, times, gDeath=X)
			# first_destiny = next((i for i, x in enumerate(Zg_des) if x), None)
			# max_cell = max(max(Zg), max(Zg_des))
			# fig, ax = plt.subplots()
			# ax.set_title(f"Clone #{counter}", x=0.01, ha='left', weight='bold', fontsize='x-large')
			# ax.set_ylabel("Cell number")
			# ax.set_xlabel("Time (hour)")

			# if first_destiny is not None:
			# 	ax.plot(times[:first_destiny+1], Zg[:first_destiny+1], 'blue', label='Dividing')
			# 	ax.plot(times[first_destiny:], Zg_des[first_destiny:], '--', c='green', label='Destiny')
			# 	ax.scatter(D, Zg_des[first_destiny], marker='*', c='green', ec='k', zorder=2.5)
			# 	ax.annotate(f"$t_{{dd}}={D:.2f}h$", xy=(D, Zg_des[first_destiny]), xycoords="data", xytext=(0.01, 0.7), textcoords='axes fraction', 
			# 				arrowprops=dict(facecolor='forestgreen', shrink=0), va='bottom', color='forestgreen', zorder=-1, fontsize='large')
			# else:
			# 	ax.plot(times, Zg, 'blue', label='Dividing')
			# 	ax.scatter(D, 0, marker='*', c='green', ec='k', zorder=2.5)  # where destiny would have been
			# 	ax.annotate(f"$t_{{dd}}={D:.2f}h$", xy=(D, 0), xycoords="data", xytext=(0.01, 0.7), textcoords='axes fraction', 
			# 				arrowprops=dict(facecolor='forestgreen', shrink=0), va='bottom', color='forestgreen', zorder=-1, fontsize='large')
			# if not np.isnan(tld):
			# 	ax.scatter(tld, max_cell, c='green', ec='k', zorder=2.5)
			# 	ax.annotate(f"$t_{{ld}}={tld:.2f}h$", xy=(tld, max_cell), xycoords="data", xytext=(0.01, 0.85), textcoords='axes fraction', 
			# 				arrowprops=dict(facecolor='forestgreen', shrink=0), va='bottom', color='forestgreen', zorder=-1, fontsize='large')
			# ax.scatter(T0, 1, c='blue', ec='k', zorder=2.5)
			# ax.scatter(X, max_cell, marker='X', c='red', ec='k', zorder=2.5)

			# # Annotate timers
			# ax.text(s=f"$t_{{div}}^0={T0:.2f}h$", x=T0+3, y=1., ha='left', va='bottom', color='blue', fontsize='large')
			# if X < tf:
			# 	ax.text(s=f"$t_{{die}}={X:.2f}h$", x=X+3, y=max_cell, ha='left', va='bottom', color='red', fontsize='large')
			# else:
			# 	ax.text(s=f"$t_{{die}}={X:.2f}h$", x=tf-35, y=max_cell, ha='left', va='bottom', color='red', fontsize='large')

			# # ax.spines['top'].set_visible(False)
			# # ax.spines['right'].set_visible(False)
			# ax.yaxis.set_major_locator(MaxNLocator(integer=True))
			# ax.set_ylim(bottom=0)
			# ax.set_xlim(left=0, right=tf)
			# fig.legend(ncol=2, frameon=False, fontsize='large')
			# fig.tight_layout(rect=(0, 0, 1, 1))
			# fig.savefig(f"{save_path_real}/c{counter}_v1.pdf", dpi=300)
			# plt.close(fig)

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
			axes[1].annotate(r"$\rightarrow$" + f"$t_{{div}}^{{k\geq1}}={T:.2f}$h", xy=(T0+1, 0.64), xycoords=('data', 'axes fraction'), fontsize='x-large')

			if np.isnan(tld):
				axes[1].scatter(tld, 1, marker='o', ec='k', c='forestgreen', zorder=10, label=f'$t_{{ld}} = $ NA')
			else:
				axes[1].scatter(tld, 1, marker='o', ec='k', c='forestgreen', zorder=10, label=f'$t_{{ld}} = {tld:.2f}$h')
			axes[1].axvline(tld, ls=':', c='forestgreen', zorder=-1)

			axes[1].scatter(D, 1, marker='*', ec='k', c='forestgreen', zorder=10, label=f'$t_{{dd}} = {D:.2f}$h')
			axes[1].axvline(D, ls=':', c='forestgreen', zorder=-1)
			axes[1].axvline(tdie, ls=':', c='red', zorder=-1)
			if tdie <= X:
				axes[1].plot([0, tdie], [1, 1], 'k', lw=2, zorder=-1)
				axes[1].scatter(tdie, 1, marker='X', ec='k', c='red', zorder=10, label=f'$t_{{die}} = {tdie:.2f}$h')
			else:
				axes[1].plot([0, tf], [1, 1], 'k', lw=2, zorder=-1)
				axes[1].scatter(tdie, 1, marker='X', ec='k', c='red', zorder=10, label=f'$t_{{die}} = {X:.2f}$h')

			axes[0].xaxis.set_visible(False)
			axes[0].spines['bottom'].set_visible(False)
			axes[1].set_xlim(left=t0, right=tf)
			axes[1].set_xlabel("Time (hour)")
			for ax in axes:
				ax.yaxis.set_visible(False)
				ax.spines['left'].set_visible(False)
			fig.legend(ncol=1, frameon=False, fontsize='x-large', handletextpad=0.01, columnspacing=0)
			fig.tight_layout(rect=(0, 0, 1, 1))
			fig.subplots_adjust(hspace=0, wspace=0)

			# fig.savefig(f"{save_path_real}/c{counter}_v2.pdf", dpi=300)
			fig.savefig(f"{save_path_real}/c{counter}.pdf", dpi=300)
			plt.close(fig)

	df_sim = pd.DataFrame({
		"tdiv0": TIME_TO_FIRST_DIV,
		"tdiv": TIME_TO_SUB_DIV,
		"tld": TIME_TO_DESTINY,
		"tdie": TIME_TO_DIE,
		"Label": LABEL
	})
	
	cols = ['#bc5090', '#003f5c']  # cols = ['#e6194B', '#003f5c']
	bin_list = np.arange(start=0, stop=np.nanmax(df_sim.max(numeric_only=True)), step=1)
	plt.rcParams.update({'axes.titlesize': 22, 'axes.labelsize': 22})
	g = sns.pairplot(df_sim, 
					 hue="Label", 
					 markers=["o", "o"], 
					 palette=sns.color_palette(cols), height=1.5, corner=True, 
					 diag_kind='hist', 
					 diag_kws=dict(bins=bin_list, density=True, ec='k', lw=1, alpha=0.3), 
					 plot_kws=dict(s=22, ec='none', linewidth=1, alpha=0.2, rasterized=True), 
					 grid_kws=dict(diag_sharey=False))
	g._legend.remove()
	g.map_diag(diag)
	titles = [r'Time to first div ($T_{div}^0$)', r'Avg. sub div time ($T_{div}^{k\geq1}$)', 
			  r'Time to last div ($T_{ld}$)', r'Time to death ($T_{die}$)']
	colors = ['blue', 'orange', 'green', 'red']
	for ax_diag, col, cor in zip(np.diag(g.axes), titles, colors):
		ax_diag.set_title(col, color=cor, fontsize=16)
	g.map_lower(corrfunc)
	plt.rcParams.update({'axes.titlesize': rc['axes.titlesize'], 
						'ytick.labelsize': rc['ytick.labelsize']})

	data_color = '#ffa600'
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
			g.axes[i,j].set_ylabel(r"$T_{div}^{k\geq 1}$", fontsize=22)
		elif i == 2 and j == 0:
			g.axes[i,j].spines['top'].set_visible(True)
			g.axes[i,j].spines['right'].set_visible(True)
			g.axes[i,j].plot(straight_x, straight_y, '--k', lw=1, zorder=-1)
			g.axes[i,j].scatter(df['tdiv0'], df['tld'], c=data_color, marker='o', ec='k', lw=0.5, s=18, rasterized=True)
			g.axes[i,j].set_ylabel(r"$T_{ld}$", fontsize=22)
		elif i == 3 and j == 0:
			g.axes[i,j].spines['top'].set_visible(True)
			g.axes[i,j].spines['right'].set_visible(True)
			g.axes[i,j].plot(straight_x, straight_y, '--k', lw=1, zorder=-1)
			g.axes[i,j].scatter(df['tdiv0'], df['tdie'], c=data_color, marker='o', ec='k', lw=0.5, s=18, rasterized=True)
			g.axes[i,j].set_xlabel(r"$T_{div}^0$", fontsize=22)
			g.axes[i,j].set_ylabel(r"$T_{die}$", fontsize=22)
			g.axes[i,j].set_xlim(left=0, right=np.nanmax(df_sim['tdiv0']))
		elif i == 2 and j == 1:
			g.axes[i,j].spines['top'].set_visible(True)
			g.axes[i,j].spines['right'].set_visible(True)
			g.axes[i,j].scatter(df['tdiv'], df['tld'], c=data_color, marker='o', ec='k', lw=0.5, s=18, rasterized=True)
		elif i == 3 and j == 1:
			g.axes[i,j].spines['top'].set_visible(True)
			g.axes[i,j].spines['right'].set_visible(True)
			g.axes[i,j].scatter(df['tdiv'], df['tdie'], c=data_color, marker='o', ec='k', lw=0.5, s=18, rasterized=True)
			g.axes[i,j].set_xlabel(r"$T_{div}^{k\geq 1}$", fontsize=22)
			g.axes[i,j].set_xlim(left=0, right=np.nanmax(df_sim['tdiv']))
		elif i == 3 and j == 2:
			g.axes[i,j].spines['top'].set_visible(True)
			g.axes[i,j].spines['right'].set_visible(True)
			g.axes[i,j].plot(straight_x, straight_y, '--k', lw=1, zorder=-1)
			g.axes[i,j].scatter(df['tld'], df['tdie'], c=data_color, marker='o', ec='k', lw=0.5, s=18, rasterized=True)
			g.axes[i,j].set_xlabel(r"$T_{ld}$", fontsize=22)
			g.axes[i,j].set_xlim(left=0, right=np.nanmax(df_sim['tld']))
	g.axes[3,3].set_xlabel(r"$T_{die}$", fontsize=22)
	handles, labels = g.axes[1,0].get_legend_handles_labels()
	leg = g.fig.legend(handles=handles, 
					   labels=labels, 
					   loc='center right',
					   markerscale=2,
					   columnspacing=1, handletextpad=0.1,
					   bbox_to_anchor=(0.9, 0.45), fontsize=16, ncol=1)  # with data (different anchor value)
	for lh in leg.legendHandles:
		lh.set_alpha(1)
	g.fig.set_size_inches(12, 9)
	g.fig.tight_layout(rect=(0, 0, 1, 1))
	g.fig.subplots_adjust(hspace=0.07, wspace=0.07)


	## Add interna plot for simulation setup
	left, bottom, width, height = [0.61, 0.63, 0.38, 0.32]
	in_ax = g.fig.add_axes([left, bottom, width, height])

	plt.rcParams.update({'ytick.labelsize': 14})
	tt = np.linspace(t0, tf, num=1000)
	in_ax.set_title(f"Cyton distribution (Fitted parameters)")
	in_ax.set_ylabel("Density", fontsize=14)
	in_ax.set_xlabel("Time (hour)", fontsize=14)
	in_ax.fill_between(tt, sps.lognorm.pdf(tt, sT0, scale=mT0), color='blue', label=f'$T_{{div}}^0 \sim LN({mT0}, {sT0})$', alpha=0.3, rasterized=True)
	in_ax.fill_between(tt, sps.lognorm.pdf(tt, sT, scale=mT), color='orange', label=f"$T_{{div}}^{{k\geq1}} \sim LN({mT}, {sT})$", alpha=0.3, rasterized=True)
	in_ax.fill_between(tt, sps.lognorm.pdf(tt, sD, scale=mD), color='green', label=f'$T_{{ld}} = T_{{dd}} \sim LN({mD}, {sD})$', alpha=0.3, rasterized=True)
	in_ax.fill_between(tt, sps.lognorm.pdf(tt, sX, scale=mX), color='red', label=f'$T_{{die}} \sim LN({mX}, {sX})$', alpha=0.3, rasterized=True)
	in_ax.spines['top'].set_visible(True)
	in_ax.spines['right'].set_visible(True)
	in_ax.set_ylim(bottom=0)
	in_ax.set_xlim(left=t0, right=tf)
	in_ax.legend(loc=1, frameon=False, fontsize=15)
	plt.rcParams.update({'ytick.labelsize': rc['ytick.labelsize']})

	g.fig.savefig(f"{save_path}/summary.pdf", dpi=300)


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
		't_misc_20140211_2.0': path_data[path_data.index(loc_data + '/' + 't_misc_20140211_2.0.csv')],
		't_misc_20140211_3.0': path_data[path_data.index(loc_data + '/' + 't_misc_20140211_3.0.csv')],
		't_misc_20140211_4.0': path_data[path_data.index(loc_data + '/' + 't_misc_20140211_4.0.csv')], 
		't_misc_20140325_1.0': path_data[path_data.index(loc_data + '/' + 't_misc_20140325_1.0.csv')], 
		't_misc_20140325_2.0': path_data[path_data.index(loc_data + '/' + 't_misc_20140325_2.0.csv')], 
		't_misc_20140325_3.0': path_data[path_data.index(loc_data + '/' + 't_misc_20140325_3.0.csv')],
		't_misc_20140325_4.0': path_data[path_data.index(loc_data + '/' + 't_misc_20140325_4.0.csv')]
	}

	pars = {  ### RECREATE FILMING DATA RESULTS (LOGNORMAL)
		'b_cpg/cpg3': {'mT0': 38.40, 'sT0': 0.13, 'mT': 10.90, 'sT': 0.23, 'mD': 53.79, 'sD': 0.21, 'mX': 86.66, 'sX': 0.18},
		'b_cpg/cpg4': {'mT0': 41.51, 'sT0': 0.14, 'mT': 12.13, 'sT': 0.24, 'mD': 58.73, 'sD': 0.22, 'mX': 86.92, 'sX': 0.22},
		't_il2/1U_aggre': {'mT0': 35.77, 'sT0': 0.12, 'mT': 18.99, 'sT': 0.38, 'mD': 37.52, 'sD': 0.16, 'mX': 44.75, 'sX': 0.23},
		't_il2/3U_aggre': {'mT0': 41.26, 'sT0': 0.15, 'mT': 17.65, 'sT': 0.50, 'mD': 46.53, 'sD': 0.21, 'mX': 65.10, 'sX': 0.28},
		't_il2/10U_aggre': {'mT0': 41.97, 'sT0': 0.18, 'mT': 16.83, 'sT': 0.19, 'mD': 47.56, 'sD': 0.21, 'mX': 63.88, 'sX': 0.27},
		't_misc_20140211_1.0': {'mT0': 34.12, 'sT0': 0.11, 'mT': 14.04, 'sT': 0.25, 'mD': 39.96, 'sD': 0.23, 'mX': 48.18, 'sX': 0.21},
		't_misc_20140211_2.0': {'mT0': 33.52, 'sT0': 0.12, 'mT': 10.0, 'sT': 0.2, 'mD': 34.19, 'sD': 0.13, 'mX': 39.53, 'sX': 0.14},
		't_misc_20140211_3.0': {'mT0': 32.85, 'sT0': 0.1, 'mT': 10.0, 'sT': 0.2, 'mD': 32.85, 'sD': 0.1, 'mX': 36.74, 'sX': 0.13},
		't_misc_20140211_4.0': {'mT0': 35.77, 'sT0': 0.1, 'mT': 16.09, 'sT': 0.2, 'mD': 37.3, 'sD': 0.14, 'mX': 43.82, 'sX': 0.19},
		't_misc_20140325_1.0': {'mT0': 34.09, 'sT0': 0.1, 'mT': 10.23, 'sT': 0.2, 'mD': 42.48, 'sD': 0.2, 'mX': 48.13, 'sX': 0.22},
		't_misc_20140325_2.0': {'mT0': 39.25, 'sT0': 0.14, 'mT': 12.83, 'sT': 0.38, 'mD': 48.38, 'sD': 0.18, 'mX': 54.22, 'sX': 0.13},
		't_misc_20140325_3.0': {'mT0': 34.5, 'sT0': 0.07, 'mT': 10.28, 'sT': 0.17, 'mD': 41.35, 'sD': 0.15, 'mX': 46.95, 'sX': 0.17},
		't_misc_20140325_4.0': {'mT0': 34.12, 'sT0': 0.08, 'mT': 11.5, 'sT': 0.23, 'mD': 39.17, 'sD': 0.19, 'mX': 43.6, 'sX': 0.18}, 
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
	