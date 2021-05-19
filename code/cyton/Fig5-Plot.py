"""
Last edit: 16-May-2021

Analyse model error with SH1.119 B cell data (9 time points, 9 replicates)
(Scenario 1) RMSE of model fits from removing the time points
(Scenario 2) Parameter errors as a function of replicate numbers
[Output] Fig5 in the main article; FigS6 in the Supplementary Material
NB: Requires fit results (in excel format) from both "Fig5AB-modelFit-Bcell.py" and "Fig5C-modelFit-Bcell.py"
"""
import sys, os, itertools
import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl; mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
import scipy.stats as sps
import multiprocessing as mp
import umap
from pca import pca  # https://github.com/erdogant/pca
from sklearn import preprocessing
from sklearn.decomposition import FastICA
import lmfit as lmf
from src.parse import parse_data
from src.utils import conf_iterval, lognorm_cdf, lognorm_pdf
from src.model import Cyton15Model

## Check library version
try:
	assert(sns.__version__=='0.10.1')
except AssertionError as ae:
	print("[VersionError] Please check if the following version of seaborn library is installed:")
	print("seaborn==0.10.1")
	sys.exit()

rng = np.random.RandomState(531236413)

rc = {
	'figure.figsize': (10, 6),
	# 'font.size': 14, 'axes.titlesize': 14, 'axes.labelsize': 12,
	# 'xtick.labelsize': 14, 'ytick.labelsize': 14,
	# 'legend.fontsize': 14, 'legend.title_fontsize': None,
	# 'axes.grid': True, 'grid.linestyle': ':', 'axes.grid.which': 'both',
	'axes.axisbelow': True, 'axes.titlepad': 0,
	'axes.spines.top': False, 'axes.spines.right': False,
	'axes.spines.left': True, 'axes.spines.bottom': True,
	'ytick.left': True, 'xtick.bottom': True,
	'lines.markersize': 7.5, 'lines.linewidth': 1.5,
	'errorbar.capsize': 4
}
sns.set(style='white', context='talk', rc=rc)

def bootstrap_cv(inputs):
	boot_df, irep, rep, df, pos, biter = inputs

	tmp = []
	# pos = mp.current_process()._identity[0]-1  # For progress bar
	pbar = tqdm.tqdm(range(biter), desc=f"Sampling for {rep} rep(s)", leave=False, position=pos+1)
	for b in pbar:
		boots = df.sample(frac=1, replace=True)
		stats = boots.drop('Replicate', axis=1).groupby(boots.Replicate).agg([sps.variation])
		tmp.append(stats)
	tmp = pd.concat(tmp)

	sub_l = boot_df[irep]
	sub_l.append(tmp)
	boot_df[irep] = sub_l

## Customise x-axis: split by group label
def add_line(ax, xpos, ypos):
	# line = plt.Line2D([xpos, xpos], [ypos+1.04, ypos], transform=ax.transAxes, color='k')
	line = plt.Line2D([xpos, xpos], [ypos+0.07, ypos], transform=ax.transAxes, color='k')
	line.set_clip_on(False)
	ax.add_line(line)

def label_len(my_index,level):
	labels = my_index.get_level_values(level)
	return [(k, sum(1 for i in g)) for k,g in itertools.groupby(labels)]

def label_group_bar_table(ax, df):
	ypos = -.04
	scale = 1./df.index.size
	for level in range(df.index.nlevels)[::-1]:
		pos = 0
		for label, rpos in label_len(df.index, level):
			lxpos = (pos + .5 * rpos)*scale
			ax.text(lxpos, ypos, label, ha='center', transform=ax.transAxes)
			add_line(ax, pos*scale, ypos)
			pos += rpos
		add_line(ax, pos*scale , ypos)
		ypos -= .1

def embed_plot(axis, combination, display, color):
	RGS = 95
	df = parse_data('./data', ['SH1.119.xlsx'])['SH1.119']
	reader = df['reader']

	fit_pars = pd.read_excel(f"{path}/data/SH1.119_{'_'.join(map(str, combination))}_result.xlsx", sheet_name='pars', index_col=0)
	fit_boots = pd.read_excel(f"{path}/data/SH1.119_{'_'.join(map(str, combination))}_result.xlsx", sheet_name='boot', index_col=0)
	icnd = 0

	### GET EXPERIMENT INFO
	hts = reader.harvested_times[icnd]
	mgen = reader.generation_per_condition[icnd]

	### PREPARE DATA
	nreps = []
	for datum in df['cgens']['rep'][icnd]:
		nreps.append(len(datum))

	pars_name = fit_pars.index.to_numpy()
	pars_vary = fit_pars.loc[:,'vary']

	best_params = lmf.Parameters()
	for par in pars_name:
		best_params.add(par, value=fit_pars.loc[par, 'mean'], vary=pars_vary[par])

	t0, tf, dt = 0, max(hts)+5, 0.5
	times = np.linspace(t0, tf, num=int(tf/dt)+1)

	### GET CYTON BEST-FIT CURVES
	model = Cyton15Model(hts, fit_pars.loc['N0', 'mean'], mgen, dt, nreps, True)
	extrapolate = model.extrapolate(times, best_params)  # get extrapolation for all "times" (discretised) and at harvested timepoints
	ext_total_live_cells = extrapolate['ext']['total_live_cells']
	ext_cells_per_gen = extrapolate['ext']['cells_gen']

	# Calculate bootstrap intervals
	b_ext_total_live_cells, b_ext_cells_per_gen = [], []

	conf = {'ext_total_live_cells': [], 'ext_cells_per_gen': []}
	tmp_N0 = []  # just to calculate confidence interval for N0, but this is not real parameter! The interval is only from bootstrapping... (And recording this would make easier to plot in the future)
	for bsample in fit_boots.drop(['algo', 'mse-in', 'mse-out', 'rmse-in', 'rmse-out'], axis=1).iterrows():
		b_mUns, b_sUns = bsample[1]['mUns'], bsample[1]['sUns']
		b_mDiv0, b_sDiv0 = bsample[1]['mDiv0'], bsample[1]['sDiv0']
		b_mDD, b_sDD = bsample[1]['mDD'], bsample[1]['sDD']
		b_mDie, b_sDie = bsample[1]['mDie'], bsample[1]['sDie']
		b_m, b_p = bsample[1]['m'], bsample[1]['p']
		b_N0 = bsample[1]['N0']

		b_params = best_params.copy()
		b_params['mUns'].set(value=b_mUns); b_params['sUns'].set(value=b_sUns)
		b_params['mDiv0'].set(value=b_mDiv0); b_params['sDiv0'].set(value=b_sDiv0)
		b_params['mDD'].set(value=b_mDD); b_params['sDD'].set(value=b_sDD)
		b_params['mDie'].set(value=b_mDie); b_params['sDie'].set(value=b_sDie)
		b_params['m'].set(value=b_m); b_params['p'].set(value=b_p)

		# Calculate model prediction for each set of parameter
		b_model = Cyton15Model(hts, b_N0, mgen, dt, nreps, True)
		b_extrapolate = b_model.extrapolate(times, b_params)  # get extrapolation for all "times" (discretised) and at harvested timepoints
		b_ext_total_live_cells.append(b_extrapolate['ext']['total_live_cells'])
		b_ext_cells_per_gen.append(b_extrapolate['ext']['cells_gen'])

		conf['ext_total_live_cells'].append(b_ext_total_live_cells)
		conf['ext_cells_per_gen'].append(b_ext_cells_per_gen)
		tmp_N0.append(b_N0)

	# Calculate 95% confidence bands on PDF, CDF and model predictions
	for obj in conf:
		stack = np.vstack(conf[obj])
		conf[obj] = conf_iterval(stack, RGS)

	cp = sns.hls_palette(mgen+1, l=0.4, s=0.5)
	hts_, total_cells_ = [], []
	hts__, total_cells__ = [], []
	for itpt, ht in enumerate(hts):
		for irep in range(nreps[itpt]):
			if itpt in combination:
				hts_.append(ht)
				total_cells_.append(df['cells']['rep'][icnd][itpt][irep])
			else:
				hts__.append(ht)
				total_cells__.append(df['cells']['rep'][icnd][itpt][irep])
	axis.plot(hts_, total_cells_, 'kx', label="Excluded")
	axis.plot(hts__, total_cells__, 'r.', label="Data")

	cgen_avg = df['cgens']['avg'][icnd]
	cgen_sem = df['cgens']['sem'][icnd]
	rmv_hts, rmv_cgen_avg, rmv_cgen_sem = [], [], []
	for itpt in sorted(combination, reverse=True):
		rmv_hts.append(hts[itpt])
		rmv_cgen_avg.append(cgen_avg[itpt])
		rmv_cgen_sem.append(cgen_sem[itpt])
		del cgen_avg[itpt]
		del cgen_sem[itpt]
		del hts[itpt]
	for igen in range(mgen+1):
		axis.scatter(rmv_hts, np.transpose(rmv_cgen_avg)[igen], color=cp[igen], marker='x')
		axis.errorbar(hts, np.transpose(cgen_avg)[igen], yerr=np.transpose(cgen_sem)[igen], color=cp[igen], fmt='.')
		axis.plot(times, ext_cells_per_gen[igen], c=cp[igen])
		axis.fill_between(times, conf['ext_cells_per_gen'][0][igen], conf['ext_cells_per_gen'][1][igen], fc=cp[igen], ec=None, alpha=0.5)

	axis.plot(times, ext_total_live_cells, 'k-', label='Model')
	axis.fill_between(times, conf['ext_total_live_cells'][0], conf['ext_total_live_cells'][1], fc='k', ec=None, alpha=0.3)
	axis.set_ylim(bottom=0)
	axis.set_xlim(left=t0, right=tf)

	axis.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	axis.yaxis.get_offset_text().set_fontsize(12)
	axis.yaxis.major.formatter._useMathText = True
	if display:
		axis.set_ylabel("Cell number", fontsize=12)
		axis.set_xlabel("Time (hour)", fontsize=12)
		axis.tick_params(axis="y", labelsize=12)
		axis.tick_params(axis="x", labelsize=12)
		axis.legend(fontsize=12, frameon=False)
	else:
		axis.set_yticks([])
		axis.set_xticks([])
		axis.set_yticklabels([])
		axis.set_xticklabels([])
	axis.set_title(f"{'-'.join(map(str, combination+1))}", fontsize=12, weight='bold', color=color)

	axis.spines['right'].set_visible(True)
	axis.spines['top'].set_visible(True)

if __name__ == "__main__":
	########################################################################################
	#								ANALYSE ACCURACY									   #
	########################################################################################
	print("> Analyse model accuracy [Handling fit results of removed time points]")

	path = './out/_lognormal/indiv/Fig5/FigC-rmvTPS'
	print(f" > Working on... {path}")
	ranges = [(0,3), (3,4), (4,6)]
	for irg, (a, b) in enumerate(ranges):
		df_fits = []
		cps = []
		tp_idx = np.arange(0, 10, step=1)
		rms = np.arange(1, 9, step=1)
		base_color = sns.color_palette(n_colors=len(rms))
		for i, rm in enumerate(rms[a:b]):
			count = 0
			removes = np.array(list(itertools.combinations(tp_idx, rm)))
			for j, comb in enumerate(removes):
				try:
					fit = pd.read_excel(f"{path}/data/SH1.119_{'_'.join(map(str, comb))}_result.xlsx", sheet_name='boot', index_col=0)
					fit.drop(['mUns', 'sUns', 'mDiv0', 'sDiv0', 'mDD', 'sDD', 'mDie', 'sDie', 'm', 'p', 'N0', 'algo'], axis=1, inplace=True)
					fit['tps'] = fit.apply(lambda x: '-'.join(map(str, comb+1)), axis=1)
					fit.index = [r"$k={0}$".format(len(comb))] * len(fit.index)
					df_fits.append(fit)
					count += 1
				except:
					pass
			print(f" >> k = -{rm}: {count} combinations")
			intermediate_color = sns.light_palette(base_color[rm-1], n_colors=count+round(0.5*count), reverse=True, input='huls')
			for c in range(count):
				cps.append(intermediate_color[c])
		df_fits = pd.concat(df_fits)
		order = []
		for lab in np.unique(df_fits.index):
			df_agg = df_fits[df_fits.index==lab].groupby('tps').agg(np.median)
			df_agg.sort_values('rmse-out', ascending=True, inplace=True)
			order.append(df_agg.index.to_list())
		order = [lab for ol in order for lab in ol]

		ref_fits = pd.read_excel(f'{path}/data/SH1.119_result.xlsx', sheet_name='boot_3reps', index_col=0)
		Q1, Q3 = ref_fits['rmse-out'].quantile(q=[0.25, 0.75])  # calculate Q1, Q3 of boxplot

		pos = 0
		fig1, ax1 = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [1,60]}, sharey=True, figsize=(18, 8))
		sns.boxplot(data=ref_fits, y='rmse-out', showfliers=True, flierprops={'marker': '.'}, color='#911eb4', ax=ax1[0])
		ax1[0].annotate(f"[Reference] All time points + 3 replicates", xy=(pos-0.23, 0.97), xycoords=('data', 'axes fraction'), va='top', weight='bold', fontsize=12, rotation=90)
		ax1[0].set_yscale('log')
		if irg < 2:  # show minor tick values
			ax1[0].yaxis.set_minor_formatter(ticker.ScalarFormatter())
			ax1[0].yaxis.minor.formatter._useMathText = True
		ax1[0].tick_params(axis='x', direction='in')
		ax1[0].set_xticklabels([])
		ax1[0].set_xlabel("")
		ax1[0].set_ylabel("RMSE (log-scale)")
		ax1[0].xaxis.tick_top()
		ax1[0].spines['top'].set_visible(True)

		pos = 0
		sns.boxplot(data=df_fits, x='tps', y='rmse-out', order=order, showfliers=True, flierprops={'marker': '.'}, palette=cps, ax=ax1[1])
		ax1[1].axhspan(ymin=Q1, ymax=Q3, color='#911eb4', alpha=0.3)
		ax1[1].axhline(y=ref_fits['rmse-out'].median(), color='#911eb4', ls='--', lw=1)
		ax1[1].tick_params(axis='x', direction='in')
		ax1[1].get_yaxis().set_visible(False)
		ax1[1].spines['left'].set_visible(False)
		label_group_bar_table(ax1[1], df_fits)
		ax1[1].set_xlabel("Position of removed time points")
		ax1[1].spines['top'].set_visible(True)
		ax1[1].xaxis.set_label_position('top') 
		ax1[1].xaxis.tick_top()
		ax1[1].grid(True, linestyle=':')
		fig1.text(0.5, 0.02, "Total number of removed time points", ha='center', va='center')
		if irg == 0:  # show minor tick values
			ax1[1].set_xticklabels(ax1[1].get_xticklabels(), weight='bold', fontsize=12, rotation=90)

			exs = [np.array([1, 6]), np.array([3, 4]), np.array([1, 5, 7]), np.array([2, 3, 4])]
			idx = []
			for ex in exs:
				string = '-'.join(map(str, ex+1))
				for index, label in enumerate(ax1[1].xaxis.get_ticklabels()):
					if string == label.get_text():
						idx.append(index)

			e1 = ax1[1].inset_axes([0.05, 0.61, 0.2, 0.32])
			embed_plot(e1, exs[0], True, 'blue')
			ax1[1].axvline(x=idx[0], color='blue', ls=':', zorder=1)
			ax1[1].xaxis.get_ticklabels()[idx[0]].set_color('blue')

			e2 = ax1[1].inset_axes([0.255, 0.61, 0.2, 0.32])
			embed_plot(e2, exs[1], False, 'red')
			ax1[1].axvline(x=idx[1], color='red', ls=':', zorder=1)
			ax1[1].get_xticklabels()[idx[1]].set_color('red')

			e3 = ax1[1].inset_axes([0.53, 0.61, 0.2, 0.32])
			embed_plot(e3, exs[2], True, 'blue')
			ax1[1].axvline(x=idx[2], color='blue', ls=':', zorder=1)
			ax1[1].xaxis.get_ticklabels()[idx[2]].set_color( 'blue')

			e4 = ax1[1].inset_axes([0.735, 0.61, 0.2, 0.32])
			embed_plot(e4, exs[3], False, 'red')
			ax1[1].axvline(x=idx[3], color='red', ls=':', zorder=1)
			ax1[1].xaxis.get_ticklabels()[idx[3]].set_color('red')
		elif irg == 1:
			ax1[1].set_xticklabels(ax1[1].get_xticklabels(), weight='bold', fontsize=13, rotation=90)

			exs = [np.array([1, 2, 3, 7]), np.array([1, 3, 5, 7]), np.array([3, 4, 5, 8]), np.array([2, 3, 4, 7])]
			idx = []
			for ex in exs:
				string = '-'.join(map(str, ex+1))
				for index, label in enumerate(ax1[1].xaxis.get_ticklabels()):
					if string == label.get_text():
						idx.append(index)

			e1 = ax1[1].inset_axes([0.055, 0.61, 0.2, 0.32])
			embed_plot(e1, exs[0], True, 'blue')
			ax1[1].axvline(x=idx[0], color='blue', ls=':', zorder=1)
			ax1[1].xaxis.get_ticklabels()[idx[0]].set_color('blue')

			e2 = ax1[1].inset_axes([0.27, 0.61, 0.2, 0.32])
			embed_plot(e2, exs[1], False, '#BAA10B')
			ax1[1].axvline(x=idx[1], color='#BAA10B', ls=':', zorder=1)
			ax1[1].get_xticklabels()[idx[1]].set_color('#BAA10B')

			e3 = ax1[1].inset_axes([0.485, 0.61, 0.2, 0.32])
			embed_plot(e3, exs[2], False, '#BAA10B')
			ax1[1].axvline(x=idx[2], color='#BAA10B', ls=':', zorder=1)
			ax1[1].xaxis.get_ticklabels()[idx[2]].set_color('#BAA10B')

			e4 = ax1[1].inset_axes([0.7, 0.61, 0.2, 0.32])
			embed_plot(e4, exs[3], False, 'red')
			ax1[1].axvline(x=idx[3], color='red', ls=':', zorder=1)
			ax1[1].xaxis.get_ticklabels()[idx[3]].set_color('red')
		else:
			ax1[1].set_xticklabels(ax1[1].get_xticklabels(), weight='bold', fontsize=11, rotation=90)

			exs = [np.array([1, 2, 4, 5, 7]), np.array([1, 2, 4, 5, 6, 7])]
			idx = []
			for ex in exs:
				string = '-'.join(map(str, ex+1))
				for index, label in enumerate(ax1[1].xaxis.get_ticklabels()):
					if string == label.get_text():
						idx.append(index)

			e1 = ax1[1].inset_axes([0.05, 0.61, 0.2, 0.32])
			embed_plot(e1, exs[0], True, 'blue')
			ax1[1].axvline(x=idx[0], color='blue', ls=':', zorder=1)
			ax1[1].xaxis.get_ticklabels()[idx[0]].set_color('blue')

			e2 = ax1[1].inset_axes([0.42, 0.61, 0.2, 0.32])
			embed_plot(e2, exs[1], True, 'blue')
			ax1[1].axvline(x=idx[1], color='blue', ls=':', zorder=1)
			ax1[1].get_xticklabels()[idx[1]].set_color('blue')
		
		fig1.tight_layout(rect=(0, 0, 1, 1))
		fig1.subplots_adjust(wspace=0.01, hspace=0)

		fig1.savefig(f"{path}/f{irg+1}_{a+1}to{b}.pdf", dpi=300)

	#########################################################################################
	#								MODEL PLOT (REPLICATE)									#
	#########################################################################################
	print("> Analyse the precision of parameter estimates [1,2,...,9 replicates]")
	RGS = 95
	df = parse_data('./data', ['SH1.119.xlsx'])['SH1.119']
	reader = df['reader']

	path = './out/_lognormal/indiv/Fig5/FigAB-allReps'
	print(f" > Working on... {path}")
	df_fits = pd.read_excel(f"{path}/SH1.119_result.xlsx", sheet_name=None, index_col=0)
	sheets = list(df_fits.keys())
	icnd = 0

	### GET EXPERIMENT INFO
	hts = reader.harvested_times[icnd]
	mgen = reader.generation_per_condition[icnd]
	condition = reader.condition_names[icnd]

	### PREPARE DATA
	nreps = []
	for datum in df['cgens']['rep'][icnd]:
		nreps.append(len(datum))

	reps = np.arange(1, 10, step=1)
	for rep in reps:
		print(f" >> {rep} replicate(s)...")
		select1 = [sheet for sheet in sheets if f"pars_{rep}reps" == sheet]
		select2 = [sheet for sheet in sheets if f"boot_{rep}reps" == sheet]

		fit_pars = df_fits[select1[0]]
		fit_boots = df_fits[select2[0]]

		### GET PARAMETER NAMES AND "VARY" STATES
		pars_name = fit_pars.index.to_numpy()
		pars_vary = fit_pars.loc[:,'vary']

		best_params = lmf.Parameters()
		for par in pars_name:
			best_params.add(par, value=fit_pars.loc[par, 'mean'], vary=pars_vary[par])
		best_fit = best_params.valuesdict()
		mUns, sUns = best_fit['mUns'], best_fit['sUns']
		mDiv0, sDiv0 = best_fit['mDiv0'], best_fit['sDiv0']
		mDD, sDD = best_fit['mDD'], best_fit['sDD']
		mDie, sDie = best_fit['mDie'], best_fit['sDie']
		m, p = best_fit['m'], best_fit['p']

		t0, tf, dt = 0, max(hts)+5, 0.5
		times = np.linspace(t0, tf, num=int(tf/dt)+1)
		gens = np.array([i for i in range(mgen+1)])

		### GET CYTON BEST-FIT CURVES
		model = Cyton15Model(hts, df['cells']['avg'][icnd][0], mgen, dt, nreps, True)
		extrapolate = model.extrapolate(times, best_params)  # get extrapolation for all "times" (discretised) and at harvested timepoints
		ext_total_live_cells = extrapolate['ext']['total_live_cells']
		ext_cells_per_gen = extrapolate['ext']['cells_gen']
		hts_total_live_cells = extrapolate['hts']['total_live_cells']
		hts_cells_per_gen = extrapolate['hts']['cells_gen']

		# Calculate bootstrap intervals
		unst_pdf_curves, unst_cdf_curves = [], []
		tdiv0_pdf_curves, tdiv0_cdf_curves = [], []
		tdd_pdf_curves, tdd_cdf_curves = [], []
		tdie_pdf_curves, tdie_cdf_curves = [], []

		b_ext_total_live_cells, b_ext_cells_per_gen = [], []
		b_hts_total_live_cells, b_hts_cells_per_gen = [], []

		conf = {
			'unst_pdf': [], 'unst_cdf': [], 'tdiv0_pdf': [], 'tdiv0_cdf': [], 'tdd_pdf': [], 'tdd_cdf': [], 'tdie_pdf': [], 'tdie_cdf': [],
			'ext_total_cohorts': [], 'ext_total_live_cells': [], 'ext_cells_per_gen': [], 'hts_total_live_cells': [], 'hts_cells_per_gen': []
		}
		tmp_N0 = []  # just to calculate confidence interval for N0, but this is not real parameter! The interval is only from bootstrapping... (And recording this would make easier to plot in the future)
		for bsample in fit_boots.drop('algo', axis=1).iterrows():
			b_mUns, b_sUns = bsample[1]['mUns'], bsample[1]['sUns']
			b_mDiv0, b_sDiv0 = bsample[1]['mDiv0'], bsample[1]['sDiv0']
			b_mDD, b_sDD = bsample[1]['mDD'], bsample[1]['sDD']
			b_mDie, b_sDie = bsample[1]['mDie'], bsample[1]['sDie']
			b_m, b_p = bsample[1]['m'], bsample[1]['p']
			b_N0 = bsample[1]['N0']

			b_params = best_params.copy()
			b_params['mUns'].set(value=b_mUns); b_params['sUns'].set(value=b_sUns)
			b_params['mDiv0'].set(value=b_mDiv0); b_params['sDiv0'].set(value=b_sDiv0)
			b_params['mDD'].set(value=b_mDD); b_params['sDD'].set(value=b_sDD)
			b_params['mDie'].set(value=b_mDie); b_params['sDie'].set(value=b_sDie)
			b_params['m'].set(value=b_m); b_params['p'].set(value=b_p)

			# Calculate PDF and CDF curves for each set of parameter
			b_unst_pdf, b_unst_cdf = lognorm_pdf(times, b_mUns, b_sUns), lognorm_cdf(times, b_mUns, b_sUns)
			b_tdiv0_pdf, b_tdiv0_cdf = lognorm_pdf(times, b_mDiv0, b_sDiv0), lognorm_cdf(times, b_mDiv0, b_sDiv0)
			b_tdd_pdf, b_tdd_cdf = lognorm_pdf(times, b_mDD, b_sDD), lognorm_cdf(times, b_mDD, b_sDD)
			b_tdie_pdf, b_tdie_cdf = lognorm_pdf(times, b_mDie, b_sDie), lognorm_cdf(times, b_mDie, b_sDie)

			unst_pdf_curves.append(b_unst_pdf); unst_cdf_curves.append(b_unst_cdf)
			tdiv0_pdf_curves.append(b_tdiv0_pdf); tdiv0_cdf_curves.append(b_tdiv0_cdf)
			tdd_pdf_curves.append(b_tdd_pdf); tdd_cdf_curves.append(b_tdd_cdf)
			tdie_pdf_curves.append(b_tdie_pdf); tdie_cdf_curves.append(b_tdie_cdf)

			# Calculate model prediction for each set of parameter
			b_model = Cyton15Model(hts, b_N0, mgen, dt, nreps, True)
			b_extrapolate = b_model.extrapolate(times, b_params)  # get extrapolation for all "times" (discretised) and at harvested timepoints
			b_ext_total_live_cells.append(b_extrapolate['ext']['total_live_cells'])
			b_ext_total_cohorts = np.sum(np.transpose(b_extrapolate['ext']['cells_gen']) * np.power(2.,-gens), axis=1)
			b_ext_cells_per_gen.append(b_extrapolate['ext']['cells_gen'])
			b_hts_total_live_cells.append(b_extrapolate['hts']['total_live_cells'])
			b_hts_cells_per_gen.append(b_extrapolate['hts']['cells_gen'])

			conf['unst_pdf'].append(b_unst_pdf); conf['unst_cdf'].append(b_unst_cdf)
			conf['tdiv0_pdf'].append(b_tdiv0_pdf); conf['tdiv0_cdf'].append(b_tdiv0_cdf)
			conf['tdd_pdf'].append(b_tdd_pdf); conf['tdd_cdf'].append(b_tdd_cdf)
			conf['tdie_pdf'].append(b_tdie_pdf); conf['tdie_cdf'].append(b_tdie_cdf)
			conf['ext_total_cohorts'].append(b_ext_total_cohorts)
			conf['ext_total_live_cells'].append(b_ext_total_live_cells); conf['ext_cells_per_gen'].append(b_ext_cells_per_gen)
			conf['hts_total_live_cells'].append(b_hts_total_live_cells); conf['hts_cells_per_gen'].append(b_hts_cells_per_gen)

			tmp_N0.append(b_N0)

		# Calculate 95% confidence bands on PDF, CDF and model predictions
		for obj in conf:
			stack = np.vstack(conf[obj])
			conf[obj] = conf_iterval(stack, RGS)
		err_mUns, err_sUns = conf_iterval(fit_boots['mUns'], RGS), conf_iterval(fit_boots['sUns'], RGS)
		err_mDiv0, err_sDiv0 = conf_iterval(fit_boots['mDiv0'], RGS), conf_iterval(fit_boots['sDiv0'], RGS)
		err_mDD, err_sDD = conf_iterval(fit_boots['mDD'], RGS), conf_iterval(fit_boots['sDD'], RGS)
		err_mDie, err_sDie = conf_iterval(fit_boots['mDie'], RGS), conf_iterval(fit_boots['sDie'], RGS)
		err_m, err_p = conf_iterval(fit_boots['m'], RGS), conf_iterval(fit_boots['p'], RGS)

		fig1, ax1 = plt.subplots(nrows=2, sharex=True, figsize=(7, 9))
		cp = sns.hls_palette(mgen+1, l=0.4, s=0.5)
		hts_, total_cells_ = [], []
		hts__, total_cells__ = [], []
		for itpt, ht in enumerate(hts):
			if rep==9:
				random = np.arange(0, 9, step=1)
			else:
				random = np.random.choice(np.arange(0, 9, step=1), size=rep, replace=False)
			for irep in range(nreps[itpt]):
				if irep not in random:
					hts_.append(ht)
					total_cells_.append(df['cells']['rep'][icnd][itpt][irep])
				else:
					hts__.append(ht)
					total_cells__.append(df['cells']['rep'][icnd][itpt][irep])
		ax1[0].plot(hts_, total_cells_, 'kx', label="Excluded")
		ax1[0].plot(hts__, total_cells__, 'ro', markersize=8, markeredgecolor='k', label=f"{rep} Random replicates")
		for igen in range(mgen+1):
			ax1[0].errorbar(hts, np.transpose(df['cgens']['avg'][icnd])[igen], yerr=np.transpose(df['cgens']['sem'][icnd])[igen], color=cp[igen], fmt='.', ms=9, label=f"Gen {igen}")
			ax1[0].plot(times, ext_cells_per_gen[igen], c=cp[igen])
			ax1[0].fill_between(times, conf['ext_cells_per_gen'][0][igen], conf['ext_cells_per_gen'][1][igen], fc=cp[igen], ec=None, alpha=0.5)

		ax1[0].set_ylabel("Cell number")
		ax1[0].plot(times, ext_total_live_cells, 'k-', label='Model')
		ax1[0].fill_between(times, conf['ext_total_live_cells'][0], conf['ext_total_live_cells'][1], fc='k', ec=None, alpha=0.3)
		ax1[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
		ax1[0].yaxis.major.formatter._useMathText = True
		ax1[0].set_ylim(bottom=0)
		handles, labels = ax1[0].get_legend_handles_labels()
		if rep == 1:
			ax1[0].legend(handles[2:], labels[2:], markerscale=1.2, columnspacing=0.5, handletextpad=0.2,  fontsize=14)
		elif rep == 3:
			ax1[0].legend(handles[:2], labels[:2], fontsize=14, columnspacing=1, handletextpad=0.1)

		ax1[1].set_title(f"$m = {m:.2f} \pm_{{{m-err_m[0]:.2f}}}^{{{err_m[1]-m:.2f}}}$", x=0.01, ha='left', fontsize=22)
		tdiv0_pdf, tdiv0_cdf = lognorm_pdf(times, mDiv0, sDiv0), lognorm_cdf(times, mDiv0, sDiv0)
		tdd_pdf, tdd_cdf = lognorm_pdf(times, mDD, sDD), lognorm_cdf(times, mDD, sDD)
		tdie_pdf, tdie_cdf = lognorm_pdf(times, mDie, sDie), lognorm_cdf(times, mDie, sDie)
		ax1[1].plot(times, tdiv0_cdf, color='blue', label=f"$T_{{div}}^0 \sim \mathcal{{LN}}({mDiv0:.2f}\pm_{{{mDiv0-err_mDiv0[0]:.2f}}}^{{{err_mDiv0[1]-mDiv0:.2f}}}, {sDiv0:.2f} \pm_{{{sDiv0-err_sDiv0[0]:.2f}}}^{{{err_sDiv0[1]-sDiv0:.2f}}})$")
		ax1[1].fill_between(times, conf['tdiv0_cdf'][0], conf['tdiv0_cdf'][1], fc='blue', ec=None, alpha=0.5)
		ax1[1].plot(times, tdd_cdf, color='green', label=f"$T_{{dd}} \sim \mathcal{{LN}}({mDD:.2f}\pm_{{{mDD-err_mDD[0]:.2f}}}^{{{err_mDD[1]-mDD:.2f}}}, {sDD:.2f}\pm_{{{sDD-err_sDD[0]:.2}}}^{{{err_sDD[1]-sDD:.2f}}})$")
		ax1[1].fill_between(times, conf['tdd_cdf'][0], conf['tdd_cdf'][1], fc='green', ec=None, alpha=0.5)
		ax1[1].plot(times, tdie_cdf, color='red', label=f"$T_{{die}} \sim \mathcal{{LN}}({mDie:.2f}\pm_{{{mDie-err_mDie[0]:.2f}}}^{{{err_mDie[1]-mDie:.2f}}}, {sDie:.2f}\pm_{{{sDie-err_sDie[0]:.2f}}}^{{{err_sDie[1]-sDie:.2f}}})$")
		ax1[1].fill_between(times, conf['tdie_cdf'][0], conf['tdie_cdf'][1], fc='red', ec=None, alpha=0.5)
		ax1[1].set_ylabel("CDF", fontsize=16)
		ax1[1].set_xlabel("Time (hour)", fontsize=16)
		ax1[1].set_ylim(bottom=0, top=1)
		ax1[1].set_xlim(left=t0, right=tf)
		ax1[1].legend(loc='upper left', markerscale=1, handletextpad=0.2, columnspacing=0.5, fontsize=18)

		fig1.tight_layout(rect=(0.0, 0.0, 1, 1))
		fig1.subplots_adjust(hspace=0.16, wspace=0)

		fig1.savefig(f"{path}/f0_{rep}reps.pdf", dpi=300)

	#########################################################################################
	#								ANALYSE PRECISION										#
	#########################################################################################
	print("> Second part of the precision analysis... [PCA, ICA, Marginal eCDFs & CVs]")
	tqdm.tqdm.set_lock(mp.RLock())

	df_fits = pd.read_excel(f'{path}/SH1.119_result.xlsx', sheet_name=None, index_col=0)
	sheets = list(df_fits.keys())

	#########################################################################################
	#								ORGANISE DATA											#
	#########################################################################################
	reps = np.arange(1, 10, step=1)
	cp = sns.color_palette("deep", len(reps))
	df_all = []
	for i, rep in enumerate(reps):
		select1 = [sheet for sheet in sheets if f"pars_{rep}reps" == sheet]
		select2 = [sheet for sheet in sheets if f"boot_{rep}reps" == sheet]
		
		fit_pars = df_fits[select1[0]]
		fit_boots = df_fits[select2[0]]
		fit_boots.drop('algo', axis=1, inplace=True)
		fit_boots['Replicate'] = rep
		df_all.append(fit_boots)
	final_df = pd.concat(df_all, ignore_index=True)

	#########################################################################################
	#							DIMENSIONALITY REDUCTION									#
	#########################################################################################
	red_df = final_df.drop(['mUns', 'sUns', 'p', 'N0', 'Replicate', 'mse-in', 'mse-out', 'rmse-in', 'rmse-out'], axis=1).copy()
	cats = final_df['Replicate'].to_numpy()
	data = red_df.to_numpy()
	data = pd.DataFrame(data, columns=[r'$m_{div}^0$', r'$s_{div}^0$', r'$m_{DD}$', r'$s_{DD}$', r'$m_{die}$', r'$s_{die}$', r'$m$'], index=cats)
	data_scaled = pd.DataFrame(preprocessing.scale(red_df), columns=[r'$m_{div}^0$', r'$s_{div}^0$', r'$m_{DD}$', r'$s_{DD}$', r'$m_{die}$', r'$s_{die}$', r'$m$'], index=cats)

	## PCA
	mpca = pca(n_components=7)
	out = mpca.fit_transform(data_scaled)
	for key, res in out.items():
		print(key)
		print(res, end='\n\n')

	## score plot / uncomment below after "Arrows variables" for biplot
	fig1, ax1 = plt.subplots(figsize=(8,6), tight_layout=True)
	red_cp = sns.color_palette('gist_rainbow', n_colors=len(reps))
	for i, rep in enumerate(reps):
		xy = out['PC'][out['PC'].index==rep]
		x, y = xy['PC1'], xy['PC2']
		ax1.scatter(x, y, color=red_cp[i], s=3, label=f"{rep}")
	fig1.legend(title="# Replicate", ncol=3, borderpad=0.3, markerscale=4, columnspacing=1, handletextpad=0.1)
	ax1.axhline(0, color='k', ls='--', zorder=0)
	ax1.axvline(0, color='k', ls='--', zorder=0)
	ax1.set_xlabel(f"PC1 ({out['explained_var'][0]*100:.1f}% explained variance)", fontsize=16)
	ax1.set_ylabel(f"PC2 ({(out['explained_var'][1]-out['explained_var'][0])*100:.1f}% explained variance)", fontsize=16)

	## Arrows variables
	y, topfeat, n_feat = mpca._fig_preprocessing(y=None, n_feat=7, d3=False)
	coeff = out['loadings'].iloc[0:n_feat, :]
	mean_x = np.mean(out['PC'].iloc[:,0].values)
	mean_y = np.mean(out['PC'].iloc[:,1].values)
	max_axis = np.max(np.abs(out['PC'].iloc[:,0:2]).min(axis=1))
	max_arrow = np.abs(coeff).max().max()
	scale = (np.max([1, np.round(max_axis / max_arrow, 2)])) * 1.25
	topfeat = topfeat.drop_duplicates(subset=['feature'])
	for i in range(0, n_feat):
		getfeat = topfeat['feature'].iloc[i]
		label = getfeat
		getcoef = coeff[getfeat].values
		xarrow = getcoef[0] * scale  # PC1 direction (aka the x-axis)
		yarrow = getcoef[1] * scale  # PC2 direction (aka the y-axis)
		txtcolor = 'k' if topfeat['type'].iloc[i] == 'weak' else 'k'
		ax1.arrow(mean_x, mean_y, xarrow - mean_x, yarrow - mean_y, color='k', width=0.005, head_width=0.02 * scale, alpha=0.8)
		ax1.text(xarrow * 1.2, yarrow * 1.2, label, color=txtcolor, ha='center', va='center')

	## Scree plot
	fig2, ax2 = plt.subplots(figsize=(8,6), tight_layout=True)
	feats = np.arange(0, len(out['explained_var']), step=1)
	rects = ax2.bar(feats, np.diff(out['explained_var'], prepend=0))
	ax2.plot(feats, out['explained_var'], 'ko-')
	ax2.axhline(0.95, color='r', ls='-')
	ax2.axvline(np.where(out['explained_var']>0.95)[0][0], color='r', ls='-')
	ax2.set_title("Cumulative explained variance\n" + f"{np.where(out['explained_var']>0.95)[0][0]+1} Principal components explain [95%] of the variance")
	ax2.set_xticks(feats)
	ax2.set_xticklabels([f"PC{i+1}" for i in range(0, len(feats))])
	ax2.set_xlabel("Principle component")
	ax2.set_ylabel("Percentage explained variance")

	## Contribution of variables to PC1 & PC2
	## http://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/112-pca-principal-component-analysis-essentials/
	eigv = out['model'].explained_variance_  # eigenvalue
	coord = out['loadings'].multiply(np.sqrt(eigv), axis=0)  # get coordinates of variables; loading = coord / sqrt(eigv)
	cos2 = coord * coord
	contrib = (cos2*100).multiply(1/cos2.sum(axis=1), axis=0)
	

	print("> Linear correlation (or coordinates)")
	tmp_df = coord.rename(columns={
		'$m_{div}^0$': 'mDiv0', '$s_{div}^0$': 'sDiv0', '$m_{DD}$': 'mDD', '$s_{DD}$': 'sDD', '$m_{die}$': 'mDie', '$s_{die}$': 'sDie', '$m$': 'm'
	})
	print(tmp_df)

	print()
	print()
	
	print("> Cos2 (i.e. squared coordinates)")
	tmp_df = cos2.rename(columns={
		'$m_{div}^0$': 'mDiv0', '$s_{div}^0$': 'sDiv0', '$m_{DD}$': 'mDD', '$s_{DD}$': 'sDD', '$m_{die}$': 'mDie', '$s_{die}$': 'sDie', '$m$': 'm'
	})
	print(tmp_df)

	print()
	print()

	print("> Contributions of vars in accounting for the variability in a PC (%)")
	tmp_df = contrib.rename(columns={
		'$m_{div}^0$': 'mDiv0', '$s_{div}^0$': 'sDiv0', '$m_{DD}$': 'mDD', '$s_{DD}$': 'sDD', '$m_{die}$': 'mDie', '$s_{die}$': 'sDie', '$m$': 'm'
	})
	print(tmp_df)


	pc1_contrib = contrib.sort_values(by='PC1', ascending=False, axis=1).loc['PC1',:]
	pc2_contrib = contrib.sort_values(by='PC2', ascending=False, axis=1).loc['PC2',:]
	def autolabel(rects, ax):
		for rect in rects:
			height = rect.get_height()
			ax.annotate(f"{height:.1f}",
						xy=(rect.get_x() + rect.get_width() / 2, height),
						xytext=(0, 3),  # 3 points vertical offset
						textcoords="offset points",
						ha='center', va='bottom', fontsize=12)		
	figg, axx = plt.subplots(ncols=2, figsize=(8,6), sharey=True)
	rects1 = axx[0].bar(pc1_contrib.index, pc1_contrib.to_numpy())
	axx[0].plot(feats, pc1_contrib.to_numpy(), 'ko-')
	axx[0].axhline(1/len(feats)*100, color='red', ls='--')  # if the contribution of the vars were uniform, the expected contribution is 1/len(vars)
	rects2 = axx[1].bar(pc2_contrib.index, pc2_contrib.to_numpy())
	axx[1].plot(feats, pc2_contrib.to_numpy(), 'ko-')
	axx[1].axhline(1/len(feats)*100, color='red', ls='--')

	axx[0].set_title("Contribution of variables\nPC1", x=0.01, ha='left')
	axx[0].set_ylabel("Contribution (%)")
	axx[1].set_title("PC2", x=0.01, ha='left')
	autolabel(rects1, axx[0])
	autolabel(rects2, axx[1])
	for axis in axx:
		for label in axis.get_xticklabels():
			label.set_rotation(90)
	figg.subplots_adjust(hspace=0, wspace=0)
	figg.tight_layout(rect=(0.0, 0.0, 1, 1))

	## ICA
	# mica = FastICA(n_components=7, random_state=rng)
	# out = mica.fit_transform(data_scaled)
	# out = pd.DataFrame(out, columns=[f'IC{i}' for i in range(1, out.shape[1]+1)], index=cats)
	# fig3, ax3 = plt.subplots(figsize=(8,6), tight_layout=True)
	# red_cp = sns.color_palette('gist_rainbow', n_colors=len(reps))
	# for i, rep in enumerate(reps):
	# 	xy = out[out.index==rep]
	# 	x, y = xy['IC1'], xy['IC2']
	# 	ax3.scatter(x, y, color=red_cp[i], s=3, label=f"{rep}")
	# fig3.legend(title="# Replicate", ncol=3, borderpad=0.3, markerscale=4, columnspacing=1, handletextpad=0.1)
	# ax3.set_xlabel(f"IC1")
	# ax3.set_ylabel(f"IC2")

	## UMAP
	# reducer = umap.UMAP()
	# embedding = reducer.fit_transform(data_scaled)
	# res = pd.DataFrame({'PC1': embedding[:,0], 'PC2': embedding[:,1]}, index=cats)
	# fig4, ax4 = plt.subplots(figsize=(8,6), tight_layout=True)
	# for i, rep in enumerate(reps):
	# 	comp = res[res.index==rep]
	# 	x, y = comp['PC1'], comp['PC2']
	# 	ax4.scatter(x, y, color=red_cp[i], s=1, label=f"{rep}")
	# ax4.spines['left'].set_visible(False)
	# ax4.spines['bottom'].set_visible(False)
	# ax4.set_yticks([])
	# ax4.set_xticks([])
	# fig4.legend(ncol=9, markerscale=4, borderpad=0.3, columnspacing=1, handletextpad=0.1)

	fig1.savefig(f'{path}/f1a_pca.pdf', dpi=300)
	fig2.savefig(f'{path}/f1b_pca.pdf', dpi=300)
	figg.savefig(f'{path}/f1c_pca.pdf', dpi=300)
	# fig3.savefig(f'{path}/f1c_ica.pdf', dpi=300)
	# fig4.savefig(f'{path}/f1d_umap.pdf', dpi=300)


	#########################################################################################
	#							MARGINAL DISTRIBUTION										#
	#########################################################################################
	# rgs = 95
	# alpha = (100 - rgs)/2
	# var_titles = [r'Time to first division ($T_{div}^0$)', r'Time to division destiny ($T_{dd}$)', r'Time to death ($T_{die}$)']
	# var_pairs = [('mDiv0', 'sDiv0'), ('mDD', 'sDD'), ('mDie', 'sDie')]
	# var_labels = [('Median (hour)', 'Shape'), ('Median (hour)', 'Shape'), ('Median (hour)', 'Shape')]
	# for i, (p1, p2) in enumerate(var_pairs):
	# 	fig1, ax1 = plt.subplots(ncols=2, sharey=True, tight_layout=True)
	# 	ax1[0].set_title(f"{var_titles[i]}", x=0, ha='left')
	# 	sns.ecdfplot(data=final_df, x=p1, hue="Replicate", palette=cp, ax=ax1[0])  # for seaborn > 0.11.0
	# 	sns.ecdfplot(data=final_df, x=p2, hue="Replicate", palette=cp, ax=ax1[1])
	# 	# for j, rep in enumerate(reps):  # for seaborn == 0.10.1
	# 	# 	x1 = final_df[final_df['Replicate']==rep][p1]
	# 	# 	x2 = final_df[final_df['Replicate']==rep][p2]
	# 	# 	sns.distplot(x1, kde=False, hist_kws=dict(cumulative=True), color=cp[j], ax=ax1[0])
	# 	# 	sns.distplot(x2, kde=False, hist_kws=dict(cumulative=True), color=cp[j], ax=ax1[1])
	# 	ax1[0].set_xlabel(var_labels[i][0])
	# 	ax1[1].set_xlabel(var_labels[i][1])
	# 	ax1[1].get_legend().remove()

	# 	g = sns.jointplot(data=final_df, x=p1, y=p2, hue="Replicate", ec='k', palette=cp, height=8, marginal_ticks=False)
	# 	g.ax_marg_x.set_title(var_titles[i], x=0, ha='left')
	# 	g.ax_joint.set_xlabel(var_labels[i][0])
	# 	g.ax_joint.set_ylabel(var_labels[i][1])
	# 	for j, rep in enumerate(reps):
	# 		df_rep = final_df[final_df['Replicate']==rep]
	# 		mean1 = df_rep[p1].mean()
	# 		low1, upp1 = df_rep[p1].quantile([alpha/100, (rgs+alpha)/100], interpolation='nearest')
	# 		mean2 = df_rep[p2].mean()
	# 		low2, upp2 = df_rep[p2].quantile([alpha/100, (rgs+alpha)/100], interpolation='nearest')
	# 		ax1[0].scatter(low1, 0.025, marker='.', ec='k', color=cp[j], zorder=10)
	# 		ax1[0].scatter(df_rep[p1].median(), 0.5, marker='.', ec='k', color=cp[j], zorder=10)
	# 		ax1[0].scatter(upp1, 0.975, marker='.', ec='k', color=cp[j], zorder=10)
	# 		ax1[0].axvspan(low1, upp1, ymin=0.025, ymax=0.975, alpha=0.4, facecolor=cp[j], edgecolor=None)

	# 		ax1[1].scatter(df_rep[p2].quantile(alpha/100), 0.025, marker='.', ec='k', color=cp[j], zorder=10)
	# 		ax1[1].scatter(df_rep[p2].median(), 0.5, marker='.', ec='k', color=cp[j], zorder=10)
	# 		ax1[1].scatter(df_rep[p2].quantile((rgs+alpha)/100), 0.975, marker='.', ec='k', color=cp[j], zorder=10)
	# 		ax1[1].axvspan(low2, upp2, ymin=0.025, ymax=0.975, alpha=0.4, facecolor=cp[j], edgecolor=None)

	# 		g.ax_joint.errorbar(mean1, mean2, xerr=[[mean1-low1], [upp1-mean1]], yerr=[[mean2-low2], [upp2-mean2]], fmt='o', mec='k', lw=2, color=cp[j])
	# 		rect = Rectangle((low1, low2), width=upp1-low1, height=upp2-low2, facecolor=cp[j], edgecolor='k', alpha=0.5)
	# 		g.ax_joint.add_patch(rect)
	# 	g.ax_joint.legend(title="# Replicate", ncol=3, borderpad=0.3, columnspacing=1, handletextpad=0.1)
	# 	fig1.savefig(f"{path}/f2a-{p1}_{p2}.pdf", dpi=300)
	# 	g.savefig(f"{path}/f2b-{p1}_{p2}.pdf", dpi=300)

	# fig2, ax2 = plt.subplots(tight_layout=True, figsize=(8, 6))
	# ax2.set_title(r"Subsequent division time ($m$)", x=0, ha='left')
	# sns.ecdfplot(data=final_df, x='m', hue="Replicate", palette=cp, ax=ax2)
	# for i, rep in enumerate(reps):
	# 	df_rep = final_df[final_df['Replicate']==rep]
	# 	mean = df_rep['m'].mean()
	# 	low, upp = df_rep['m'].quantile([alpha/100, (95+alpha)/100], interpolation='nearest')
	# 	# ax2.axvline(df_rep['m'].mean(), ls='--', color=cp[i])
	# 	ax2.scatter(low, 0.025, marker='.', ec='k', color=cp[i], zorder=10)
	# 	ax2.scatter(df_rep['m'].median(), 0.5, marker='.', ec='k', color=cp[i], zorder=10)
	# 	ax2.scatter(upp, 0.975, marker='.', ec='k', color=cp[i], zorder=10)
	# 	ax2.axvspan(low, upp, ymin=0.025, ymax=0.975, alpha=0.4, facecolor=cp[i], edgecolor=None)
	# ax2.set_xlabel("Time (hour)")
	# fig2.savefig(f"{path}/f2c_m.pdf", dpi=300)

	#########################################################################################
	#							COEFFICIENT OF VARIATION									#
	#########################################################################################
	inputs = []
	manager = mp.Manager()
	boot_df = manager.list([[]]*len(reps))

	# Run bootstrap for 95% CI for each CV
	cv_df = final_df.drop(['mUns', 'sUns', 'p', 'N0'], axis=1).copy()
	for irep, rep in enumerate(reps):
		df = cv_df[cv_df['Replicate']==rep]
		inputs.append((boot_df, irep, rep, df, irep+1, 1000000))  # set number of bootstrap samples
	with mp.Pool(initializer=tqdm.tqdm.set_lock, initargs=(tqdm.tqdm.get_lock(),)) as p:
		with tqdm.tqdm(total=len(inputs), desc="Total", position=0) as pbar:
			for i, _ in enumerate(p.imap_unordered(bootstrap_cv, inputs)):
				pbar.update()

	boot_df = [ll for l in boot_df for ll in l]  # flatten the shared list
	boot_df = pd.concat(boot_df)
	quants = boot_df.groupby(boot_df.index).quantile([0.025, 0.975], interpolation='nearest')
	stats = final_df.drop(['mUns', 'sUns', 'p', 'N0', 'Replicate'], axis=1).groupby(final_df.Replicate).agg([sps.variation])

	idx = pd.IndexSlice
	x = stats.index.to_numpy(dtype=str)
	for i, l in enumerate(x):
		x[i] = f'+{l}'

	fig3, ax3 = plt.subplots(nrows=2, ncols=2, figsize=(12,8))
	ax3[0,0].set_title("Median")
	ax3[0,0].errorbar(stats.index, stats['mDiv0']['variation'], 
						yerr=[stats['mDiv0']['variation']-quants.loc[idx[:,0.025], :]['mDiv0']['variation'], quants.loc[idx[:,0.975], :]['mDiv0']['variation']-stats['mDiv0']['variation']], fmt='o-', ms=6, color='b', label=r"$T_{div}^0$")
	ax3[0,0].errorbar(stats.index, stats['mDD']['variation'], 
						yerr=[stats['mDD']['variation']-quants.loc[idx[:,0.025], :]['mDD']['variation'], quants.loc[idx[:,0.975], :]['mDD']['variation']-stats['mDD']['variation']], fmt='o-', ms=6, color='g', label=r"$T_{dd}$")
	ax3[0,0].errorbar(stats.index, stats['mDie']['variation'], 
						yerr=[stats['mDie']['variation']-quants.loc[idx[:,0.025], :]['mDie']['variation'], quants.loc[idx[:,0.975], :]['mDie']['variation']-stats['mDie']['variation']], fmt='o-', ms=6, color='r', label=r"$T_{die}$")
	ax3[0,0].errorbar(stats.index, stats['m']['variation'], 
						yerr=[stats['m']['variation']-quants.loc[idx[:,0.025], :]['m']['variation'], quants.loc[idx[:,0.975], :]['m']['variation']-stats['m']['variation']], fmt='o-', ms=6, color='navy', label=r"$m$")
	ax3[0,0].xaxis.set_major_locator(MaxNLocator(integer=True))
	ax3[0,0].set_ylabel("CV", fontsize=20)
	ax3[0,0].set_xlabel("# Replicate", fontsize=20)
	ax3[0,0].set_ylim(bottom=0)
	# ax3[0,0].set_yscale('log')
	ax3[0,0].legend()

	ax3[1,0].axhline(0, ls='--', c='k')
	ax3[1,0].errorbar(x[:-1], np.diff(stats['mDiv0']['variation']), 
						yerr=[np.diff(stats['mDiv0']['variation']-quants.loc[idx[:,0.025], :]['mDiv0']['variation']), np.diff(quants.loc[idx[:,0.975], :]['mDiv0']['variation']-stats['mDiv0']['variation'])], fmt='o-', ms=6, color='b', label=r"$T_{div}^0$")
	ax3[1,0].errorbar(x[:-1], np.diff(stats['mDD']['variation']), 
						yerr=[np.diff(stats['mDD']['variation']-quants.loc[idx[:,0.025], :]['mDD']['variation']), np.diff(quants.loc[idx[:,0.975], :]['mDD']['variation']-stats['mDD']['variation'])], fmt='o-', ms=6, color='g', label=r"$T_{dd}$")
	ax3[1,0].errorbar(x[:-1], np.diff(stats['mDie']['variation']), 
						yerr=[np.diff(stats['mDie']['variation']-quants.loc[idx[:,0.025], :]['mDie']['variation']), np.diff(quants.loc[idx[:,0.975], :]['mDie']['variation']-stats['mDie']['variation'])], fmt='o-', ms=6, color='r', label=r"$T_{die}$")
	ax3[1,0].errorbar(x[:-1], np.diff(stats['m']['variation']), 
						yerr=[np.diff(stats['m']['variation']-quants.loc[idx[:,0.025], :]['m']['variation']), np.diff(quants.loc[idx[:,0.975], :]['m']['variation']-stats['m']['variation'])], fmt='o-', ms=6, color='navy', label=r"$m$")
	ax3[1,0].xaxis.set_major_locator(MaxNLocator(integer=True))
	ax3[1,0].set_ylabel(r"$\Delta$CV", fontsize=20)
	ax3[1,0].set_xlabel(r"$\Delta$#Replicate", fontsize=20)


	ax3[0,1].set_title("Shape")
	ax3[0,1].errorbar(stats.index, stats['sDiv0']['variation'], 
						yerr=[stats['sDiv0']['variation']-quants.loc[idx[:,0.025], :]['sDiv0']['variation'], quants.loc[idx[:,0.975], :]['sDiv0']['variation']-stats['sDiv0']['variation']], fmt='o-', ms=6, color='b', label=r"$T_{div}^0$")
	ax3[0,1].errorbar(stats.index, stats['sDD']['variation'], 
						yerr=[stats['sDD']['variation']-quants.loc[idx[:,0.025], :]['sDD']['variation'], quants.loc[idx[:,0.975], :]['sDD']['variation']-stats['sDD']['variation']], fmt='o-', ms=6, color='g', label=r"$T_{dd}$")
	ax3[0,1].errorbar(stats.index, stats['sDie']['variation'], 
						yerr=[stats['sDie']['variation']-quants.loc[idx[:,0.025], :]['sDie']['variation'], quants.loc[idx[:,0.975], :]['sDie']['variation']-stats['sDie']['variation']], fmt='o-', ms=6, color='r', label=r"$T_{die}$")
	ax3[0,1].xaxis.set_major_locator(MaxNLocator(integer=True))
	ax3[0,1].set_xlabel("# Replicate", fontsize=20)
	ax3[0,1].set_ylim(bottom=0)
	# ax3[0,1].set_yscale('log')
	ax3[0,1].legend()

	ax3[1,1].axhline(0, ls='--', c='k')
	ax3[1,1].errorbar(x[:-1], np.diff(stats['sDiv0']['variation']), 
						yerr=[np.diff(stats['sDiv0']['variation']-quants.loc[idx[:,0.025], :]['sDiv0']['variation']), np.diff(quants.loc[idx[:,0.975], :]['sDiv0']['variation']-stats['sDiv0']['variation'])], fmt='o-', ms=6, color='b', label=r"$T_{div}^0$")
	ax3[1,1].errorbar(x[:-1], np.diff(stats['sDD']['variation']), 
						yerr=[np.diff(stats['sDD']['variation']-quants.loc[idx[:,0.025], :]['sDD']['variation']), np.diff(quants.loc[idx[:,0.975], :]['sDD']['variation']-stats['sDD']['variation'])], fmt='o-', ms=6, color='g', label=r"$T_{dd}$")
	ax3[1,1].errorbar(x[:-1], np.diff(stats['sDie']['variation']), 
						yerr=[np.diff(stats['sDie']['variation']-quants.loc[idx[:,0.025], :]['sDie']['variation']), np.diff(quants.loc[idx[:,0.975], :]['sDie']['variation']-stats['sDie']['variation'])], fmt='o-', ms=6, color='r', label=r"$T_{die}$")
	ax3[1,1].xaxis.set_major_locator(MaxNLocator(integer=True))
	ax3[1,1].set_xlabel(f"$\Delta$#Replicate", fontsize=20)

	fig3.tight_layout(rect=(0, 0, 1, 1))
	fig3.subplots_adjust(hspace=0.25, wspace=0.15)
	fig3.savefig(f"{path}/f3_cv.pdf", dpi=300)