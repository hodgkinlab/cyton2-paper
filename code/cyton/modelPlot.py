"""
Last edit: 17-Oct-2020
Plot tool for JM-Science Fig3 results. [EX127, EX130b]
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lmfit as lmf
# pd.set_option('display.max_rows', 999999)

from src.parse import parse_data
from src.utils import conf_iterval, lognorm_cdf, lognorm_pdf, norm_cdf, norm_pdf, lognorm_statistics
from src.model import Cyton15Model

rc = {
	# 'font.size': 14, 'axes.titlesize': 14, 'axes.labelsize': 12,
	# 'xtick.labelsize': 14, 'ytick.labelsize': 14,
	'figure.figsize': (10, 8),
	# 'axes.grid': True, 'grid.linestyle': ':', 'axes.grid.which': 'both',
	'axes.axisbelow': True, 'axes.titlepad': 0,
	'axes.spines.top': False, 'axes.spines.right': False,
	'axes.spines.left': True, 'axes.spines.bottom': True,
	'ytick.left': True, 'xtick.bottom': True,
	'lines.markersize': 7.5, 'lines.linewidth': 1.5,
	'errorbar.capsize': 4
}
sns.set(style='white', context='talk', rc=rc)

PATH_TO_DATA = './data'
DATA = ["EX127.xlsx", "EX130b.xlsx"]
KEYS = [os.path.splitext(os.path.basename(data_key))[0] for data_key in DATA]
""" DATA DESCRIPTION
EX127: Fig.3 of Marchingo et al. Science 2014
	(1) N4, (2) aCD27, (3) aCD28, (4) IL12, (5) aCD27+IL12, (6) aCD27+aCD28, (7) aCD28+IL12, (8) aCD27+aCD28+IL12
EX130b: repeat of EX127
"""
RGS = 95
df_data = parse_data(PATH_TO_DATA, DATA)

for key in KEYS:
	df = df_data[key]
	df_fits = pd.read_excel(f"./out/_all_Normal/joint/siter200/{key}_result.xlsx", sheet_name=None, index_col=0)
	sheets = list(df_fits.keys())

	reader = df['reader']
	cp = sns.color_palette("deep", len(reader.condition_names)); max_time = 0
	fig1, ax1 = plt.subplots()  
	fig2, ax2 = plt.subplots(nrows=3, ncols=1, sharex=True)  # CDFs of the parameters
	count = 1
	for icnd, cond in enumerate(reader.condition_names):
		if cond == 'IL-12' or cond == 'aCD27 + IL-12' or cond == 'aCD28 + IL-12' or cond == 'aCD27 + aCD28 + IL-12':
			continue
		
		### GET EXPERIMENT INFO
		hts = reader.harvested_times[icnd]
		mgen = reader.generation_per_condition[icnd]
		condition = reader.condition_names[icnd]

		### PREPARE DATA
		data = df['cgens']['rep'][icnd]; nreps = []
		for datum in data:
			nreps.append(len(datum))
		ax1.errorbar(hts, df['cells']['avg'][icnd], yerr=df['cells']['sem'][icnd], fmt='o', color=cp[icnd], label=f"{''.join(condition.split())}")

		if cond == 'N4' or cond == 'aCD27' or cond == 'aCD28':
			### SELECT EXCEL SHEETS
			select1 = [sheet for sheet in sheets if f"pars_{cond}" == sheet]
			select2 = [sheet for sheet in sheets if f"boot_{cond}" == sheet]
			
			fit_pars = df_fits[select1[0]]
			fit_boots = df_fits[select2[0]]

			### GET PARAMETER NAMES AND "VARY" STATES
			pars_name = fit_pars.index.to_numpy()
			pars_vary = fit_pars.loc[:,'vary']

			best_params = lmf.Parameters()
			for par in pars_name:
				best_params.add(par, value=fit_pars.loc[par, 'best-fit'], vary=pars_vary[par])
			best_fit = best_params.valuesdict()
			mUns, sUns = best_fit['mUns'], best_fit['sUns']
			mDiv0, sDiv0 = best_fit['mDiv0'], best_fit['sDiv0']
			mDD, sDD = best_fit['mDD'], best_fit['sDD']
			mDie, sDie = best_fit['mDie'], best_fit['sDie']
			m, p = best_fit['m'], best_fit['p']

			t0, tf, dt = 0, max(hts)+5, 0.1
			if tf > max_time: max_time = tf
			times = np.linspace(t0, tf, num=int(tf/dt)+1)
			gens = np.array([i for i in range(mgen+1)])

			### GET CYTON BEST-FIT CURVES
			model = Cyton15Model(hts, df['cells']['avg'][icnd][0], mgen, dt, nreps, False)
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
			cols = fit_boots.columns[fit_boots.columns.to_series().str.contains(f'_{icnd}')].append(pd.Index(['m']))
			for bsample in fit_boots[cols].iterrows():
				b_mUns, b_sUns = bsample[1][f'mUns_{icnd}'], bsample[1][f'sUns_{icnd}']
				b_mDiv0, b_sDiv0 = bsample[1][f'mDiv0_{icnd}'], bsample[1][f'sDiv0_{icnd}']
				b_mDD, b_sDD = bsample[1][f'mDD_{icnd}'], bsample[1][f'sDD_{icnd}']
				b_mDie, b_sDie = bsample[1][f'mDie_{icnd}'], bsample[1][f'sDie_{icnd}']
				b_m, b_p = bsample[1]['m'], bsample[1][f'p_{icnd}']
				b_N0 = bsample[1][f'N0_{icnd}']

				b_params = best_params.copy()
				b_params['mUns'].set(value=b_mUns); b_params['sUns'].set(value=b_sUns)
				b_params['mDiv0'].set(value=b_mDiv0); b_params['sDiv0'].set(value=b_sDiv0)
				b_params['mDD'].set(value=b_mDD); b_params['sDD'].set(value=b_sDD)
				b_params['mDie'].set(value=b_mDie); b_params['sDie'].set(value=b_sDie)
				b_params['m'].set(value=b_m); b_params['p'].set(value=b_p)

				# Calculate PDF and CDF curves for each set of parameter
				b_unst_pdf, b_unst_cdf = norm_pdf(times, b_mUns, b_sUns), norm_cdf(times, b_mUns, b_sUns)
				b_tdiv0_pdf, b_tdiv0_cdf = norm_pdf(times, b_mDiv0, b_sDiv0), norm_cdf(times, b_mDiv0, b_sDiv0)
				b_tdd_pdf, b_tdd_cdf = norm_pdf(times, b_mDD, b_sDD), norm_cdf(times, b_mDD, b_sDD)
				b_tdie_pdf, b_tdie_cdf = norm_pdf(times, b_mDie, b_sDie), norm_cdf(times, b_mDie, b_sDie)

				unst_pdf_curves.append(b_unst_pdf); unst_cdf_curves.append(b_unst_cdf)
				tdiv0_pdf_curves.append(b_tdiv0_pdf); tdiv0_cdf_curves.append(b_tdiv0_cdf)
				tdd_pdf_curves.append(b_tdd_pdf); tdd_cdf_curves.append(b_tdd_cdf)
				tdie_pdf_curves.append(b_tdie_pdf); tdie_cdf_curves.append(b_tdie_cdf)

				# Calculate model prediction for each set of parameter
				b_model = Cyton15Model(hts, b_N0, mgen, dt, nreps, False)
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
			err_mUns, err_sUns = conf_iterval(fit_boots[f'mUns_{icnd}'], RGS), conf_iterval(fit_boots[f'sUns_{icnd}'], RGS)
			err_mDiv0, err_sDiv0 = conf_iterval(fit_boots[f'mDiv0_{icnd}'], RGS), conf_iterval(fit_boots[f'sDiv0_{icnd}'], RGS)
			err_mDD, err_sDD = conf_iterval(fit_boots[f'mDD_{icnd}'], RGS), conf_iterval(fit_boots[f'sDD_{icnd}'], RGS)
			err_mDie, err_sDie = conf_iterval(fit_boots[f'mDie_{icnd}'], RGS), conf_iterval(fit_boots[f'sDie_{icnd}'], RGS)
			err_m, err_p = conf_iterval(fit_boots['m'], RGS), conf_iterval(fit_boots[f'p_{icnd}'], RGS)

			ax1.set_ylabel("Cell number")
			ax1.set_xlabel("Time (hour)")
			ax1.plot(times, ext_total_live_cells, color=cp[icnd])
			ax1.fill_between(times, conf['ext_total_live_cells'][0], conf['ext_total_live_cells'][1], fc=cp[icnd], ec=None, alpha=0.5)
			# ax1.scatter(hts, np.array(df['cells']['avg'][icnd])/df['cells']['avg'][icnd][0], color=cp[icnd], label=f"{condition}")
			# ax1.plot(times, ext_total_live_cells/ext_total_live_cells[0], color=cp[icnd])

			tdiv0_pdf, tdiv0_cdf = norm_pdf(times, mDiv0, sDiv0), norm_cdf(times, mDiv0, sDiv0)
			tdd_pdf, tdd_cdf = norm_pdf(times, mDD, sDD), norm_cdf(times, mDD, sDD)
			tdie_pdf, tdie_cdf = norm_pdf(times, mDie, sDie), norm_cdf(times, mDie, sDie)
			ax2[0].plot(times, tdiv0_cdf, color=cp[icnd], label=f"$\mathcal{{N}}({mDiv0:.2f}\pm_{{{mDiv0-err_mDiv0[0]:.2f}}}^{{{err_mDiv0[1]-mDiv0:.2f}}}, {sDiv0:.2f} \pm_{{{sDiv0-err_sDiv0[0]:.2f}}}^{{{err_sDiv0[1]-sDiv0:.2f}}})$")
			ax2[0].fill_between(times, conf['tdiv0_cdf'][0], conf['tdiv0_cdf'][1], fc=cp[icnd], ec=None, alpha=0.5)
			ax2[1].plot(times, tdd_cdf, color=cp[icnd], label=f"$\mathcal{{N}}({mDD:.2f}\pm_{{{mDD-err_mDD[0]:.2f}}}^{{{err_mDD[1]-mDD:.2f}}}, {sDD:.2f}\pm_{{{sDD-err_sDD[0]:.2}}}^{{{err_sDD[1]-sDD:.2f}}})$")
			ax2[1].fill_between(times, conf['tdd_cdf'][0], conf['tdd_cdf'][1], fc=cp[icnd], ec=None, alpha=0.5)
			ax2[2].plot(times, tdie_cdf, color=cp[icnd], label=f"$\mathcal{{N}}({mDie:.2f}\pm_{{{mDie-err_mDie[0]:.2f}}}^{{{err_mDie[1]-mDie:.2f}}}, {sDie:.2f}\pm_{{{sDie-err_sDie[0]:.2f}}}^{{{err_sDie[1]-sDie:.2f}}})$")
			ax2[2].fill_between(times, conf['tdie_cdf'][0], conf['tdie_cdf'][1], fc=cp[icnd], ec=None, alpha=0.5)

			# ax2[0].annotate(f"[{''.join(condition.split())}]: $m = {m:.2f}\pm_{{{m-err_m[0]:.2f}}}^{{{err_m[1]-m:.2f}}}$", xy=(1, 0.98-count/8), color=cp[icnd], fontsize=14)
			# count += 1

	### PREDICTION LINE
	icnd = 5
	lab = 'aCD27+aCD28'

	### GET EXPERIMENT INFO
	hts = reader.harvested_times[icnd]
	mgen = reader.generation_per_condition[icnd]
	data = df['cgens']['rep'][icnd]; nreps = []
	for datum in data:
		nreps.append(len(datum))

	pred_pars = df_fits['pars_pred_aCD27_aCD28']
	pars_name = pred_pars.index.to_numpy()
	pred_params = lmf.Parameters()
	for par in pars_name:
		pred_params.add(par, value=pred_pars.loc[par, 'best-fit'])

	## Calculate bootstrap intervals
	unst_pdf_curves, unst_cdf_curves = [], []
	tdiv0_pdf_curves, tdiv0_cdf_curves = [], []
	tdd_pdf_curves, tdd_cdf_curves = [], []
	tdie_pdf_curves, tdie_cdf_curves = [], []

	pb_ext_total_live_cells, pb_ext_cells_per_gen = [], []
	pb_hts_total_live_cells, pb_hts_cells_per_gen = [], []
	conf = {
		'unst_pdf': [], 'unst_cdf': [], 'tdiv0_pdf': [], 'tdiv0_cdf': [], 'tdd_pdf': [], 'tdd_cdf': [], 'tdie_pdf': [], 'tdie_cdf': [],
		'ext_total_cohorts': [], 'ext_total_live_cells': [], 'ext_cells_per_gen': [], 'hts_total_live_cells': [], 'hts_cells_per_gen': []
	}
	pred_boots = df_fits['boot_pred_aCD27_aCD28'].dropna(axis=0)
	for pbsample in pred_boots.iterrows():
		pb_mUns, pb_sUns, pb_mDiv0, pb_sDiv0, pb_mDD, pb_sDD, pb_mDie, pb_sDie, pb_m, pb_p = pbsample[1].values
		pb_params = pred_params.copy()
		pb_params['mUns'].set(value=pb_mUns); pb_params['sUns'].set(value=pb_sUns)
		pb_params['mDiv0'].set(value=pb_mDiv0); pb_params['sDiv0'].set(value=pb_sDiv0)
		pb_params['mDD'].set(value=pb_mDD); pb_params['sDD'].set(value=pb_sDD)
		pb_params['mDie'].set(value=pb_mDie); pb_params['sDie'].set(value=pb_sDie)
		pb_params['m'].set(value=pb_m); pb_params['p'].set(value=pb_p)
		pb_N0 = df['cells']['avg'][icnd][0]

		# Calculate PDF and CDF curves for each set of parameter
		pb_unst_pdf, pb_unst_cdf = norm_pdf(times, pb_mUns, pb_sUns), norm_cdf(times, pb_mUns, pb_sUns)
		pb_tdiv0_pdf, pb_tdiv0_cdf = norm_pdf(times, pb_mDiv0, pb_sDiv0), norm_cdf(times, pb_mDiv0, pb_sDiv0)
		pb_tdd_pdf, pb_tdd_cdf = norm_pdf(times, pb_mDD, pb_sDD), norm_cdf(times, pb_mDD, pb_sDD)
		pb_tdie_pdf, pb_tdie_cdf = norm_pdf(times, pb_mDie, pb_sDie), norm_cdf(times, pb_mDie, pb_sDie)

		unst_pdf_curves.append(pb_unst_pdf); unst_cdf_curves.append(pb_unst_cdf)
		tdiv0_pdf_curves.append(pb_tdiv0_pdf); tdiv0_cdf_curves.append(pb_tdiv0_cdf)
		tdd_pdf_curves.append(pb_tdd_pdf); tdd_cdf_curves.append(pb_tdd_cdf)
		tdie_pdf_curves.append(pb_tdie_pdf); tdie_cdf_curves.append(pb_tdie_cdf)

		# Calculate model prediction for each set of parameter
		pb_model = Cyton15Model(hts, pb_N0, mgen, dt, nreps, False)
		pb_extrapolate = pb_model.extrapolate(times, pb_params)  # get extrapolation for all "times" (discretised) and at harvested timepoints
		pb_ext_total_live_cells.append(pb_extrapolate['ext']['total_live_cells'])
		pb_ext_total_cohorts = np.sum(np.transpose(pb_extrapolate['ext']['cells_gen']) * np.power(2.,-gens), axis=1)
		pb_ext_cells_per_gen.append(pb_extrapolate['ext']['cells_gen'])
		pb_hts_total_live_cells.append(pb_extrapolate['hts']['total_live_cells'])
		pb_hts_cells_per_gen.append(pb_extrapolate['hts']['cells_gen'])

		conf['unst_pdf'].append(pb_unst_pdf); conf['unst_cdf'].append(pb_unst_cdf)
		conf['tdiv0_pdf'].append(pb_tdiv0_pdf); conf['tdiv0_cdf'].append(pb_tdiv0_cdf)
		conf['tdd_pdf'].append(pb_tdd_pdf); conf['tdd_cdf'].append(pb_tdd_cdf)
		conf['tdie_pdf'].append(pb_tdie_pdf); conf['tdie_cdf'].append(pb_tdie_cdf)
		conf['ext_total_cohorts'].append(pb_ext_total_cohorts)
		conf['ext_total_live_cells'].append(pb_ext_total_live_cells); conf['ext_cells_per_gen'].append(pb_ext_cells_per_gen)
		conf['hts_total_live_cells'].append(pb_hts_total_live_cells); conf['hts_cells_per_gen'].append(pb_hts_cells_per_gen)
	for obj in conf:
		stack = np.vstack(conf[obj])
		conf[obj] = conf_iterval(stack, RGS)
	err_mUns, err_sUns = conf_iterval(pred_boots['mUns'], RGS), conf_iterval(pred_boots['sUns'], RGS)
	err_mDiv0, err_sDiv0 = conf_iterval(pred_boots['mDiv0'], RGS), conf_iterval(pred_boots['sDiv0'], RGS)
	err_mDD, err_sDD = conf_iterval(pred_boots['mDD'], RGS), conf_iterval(pred_boots['sDD'], RGS)
	err_mDie, err_sDie = conf_iterval(pred_boots['mDie'], RGS), conf_iterval(pred_boots['sDie'], RGS)
	err_m, err_p = conf_iterval(pred_boots['m'], RGS), conf_iterval(pred_boots['p'], RGS)

	tdiv0_pdf, tdiv0_cdf = norm_pdf(times, pred_params['mDiv0'], pred_params['sDiv0']), norm_cdf(times, pred_params['mDiv0'], pred_params['sDiv0'])
	tdd_pdf, tdd_cdf = norm_pdf(times, pred_params['mDD'], pred_params['sDD']), norm_cdf(times, pred_params['mDD'], pred_params['sDD'])
	tdie_pdf, tdie_cdf = norm_pdf(times, pred_params['mDie'], pred_params['sDie']), norm_cdf(times, pred_params['mDie'], pred_params['sDie'])
	m = pred_pars.loc['m', 'best-fit']
	ax2[0].annotate(f"$m = {m:.2f}\pm_{{{m-err_m[0]:.2f}}}^{{{err_m[1]-m:.2f}}}$", xy=(1, 0.85), color='k', fontsize=16)
	ax2[0].plot(times, tdiv0_cdf, '--', color=cp[icnd], 
				label=f"$\mathcal{{N}}({pred_params['mDiv0'].value:.2f}\pm_{{{pred_params['mDiv0'].value-err_mDiv0[0]:.2f}}}^{{{err_mDiv0[1]-pred_params['mDiv0'].value:.2f}}}, {pred_params['sDiv0'].value:.2f} \pm_{{{pred_params['sDiv0'].value-err_sDiv0[0]:.2f}}}^{{{err_sDiv0[1]-pred_params['sDiv0'].value:.2f}}})$")
	ax2[0].fill_between(times, conf['tdiv0_cdf'][0], conf['tdiv0_cdf'][1], fc=cp[icnd], ec=None, alpha=0.5)
	ax2[1].plot(times, tdd_cdf, '--', color=cp[icnd], 
				label=f"$\mathcal{{N}}({pred_params['mDD'].value:.2f}\pm_{{{pred_params['mDD'].value-err_mDD[0]:.2f}}}^{{{err_mDD[1]-pred_params['mDD'].value:.2f}}}, {pred_params['sDD'].value:.2f}\pm_{{{pred_params['sDD'].value-err_sDD[0]:.2f}}}^{{{err_sDD[1]-pred_params['sDD'].value:.2f}}})$")
	ax2[1].fill_between(times, conf['tdd_cdf'][0], conf['tdd_cdf'][1], fc=cp[icnd], ec=None, alpha=0.5)
	ax2[2].plot(times, tdie_cdf, '--', color=cp[icnd], 
				label=f"$\mathcal{{N}}({pred_params['mDie'].value:.2f}\pm_{{{pred_params['mDie'].value-err_mDie[0]:.2f}}}^{{{err_mDie[1]-pred_params['mDie'].value:.2f}}}, {pred_params['sDie'].value:.2f}\pm_{{{pred_params['sDie'].value-err_sDie[0]:.2f}}}^{{{err_sDie[1]-pred_params['sDie'].value:.2f}}})$")
	ax2[2].fill_between(times, conf['tdie_cdf'][0], conf['tdie_cdf'][1], fc=cp[icnd], ec=None, alpha=0.5)

	pred_model = Cyton15Model(hts, df['cells']['avg'][icnd][0], mgen, dt, nreps, False)
	extrapolate = pred_model.extrapolate(times, pred_params)
	ax1.plot(times, extrapolate['ext']['total_live_cells'], '--', color=cp[icnd], label='Predicted')
	ax1.fill_between(times, conf['ext_total_live_cells'][0], conf['ext_total_live_cells'][1], fc=cp[icnd], ec=None, alpha=0.5)

	handles, labels = ax1.get_legend_handles_labels()
	handles = handles[1:] + handles[0:1]
	handles.reverse()
	labels = labels[1:] + labels[0:1]
	labels.reverse()
	ax1.legend(labels=labels, handles=handles, frameon=True, fontsize=20)
	ax1.set_xlim(left=0, right=max_time)
	# ax1.set_yscale('log')
	# ax1.set_ylim(bottom=1E3)
	ax1.set_ylim(bottom=0)
	ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	ax1.yaxis.major.formatter._useMathText = True
	ax2_titles = ["Time to first division ($T_{div}^0$)", "Time to division destiny ($T_{dd}$)", "Time to death ($T_{die}$)"]
	for ix, axis in enumerate(ax2):
		axis.set_title(ax2_titles[ix], x=0.01, ha='left')
		axis.set_ylabel("CDF")
		axis.set_ylim(bottom=0, top=1)
		axis.set_xlim(left=0, right=max_time)
		handles, labels = axis.get_legend_handles_labels()
		handles.reverse()
		labels.reverse()
		axis.legend(ncol=1, handles=handles, labels=labels, frameon=False, fontsize=16)
	ax2[2].set_xlabel("Time (hour)")
	fig1.tight_layout(rect=(0, 0, 1, 1))
	fig2.tight_layout(rect=(0, 0, 1, 1))
	fig2.subplots_adjust(wspace=0, hspace=0.2)

	fig1.savefig(f"./out/_all_Normal/joint/_prediction/{key}_cells.pdf", dpi=300)
	fig2.savefig(f"./out/_all_Normal/joint/_prediction/{key}_params.pdf", dpi=300)

	# plt.show()














# for key in KEYS:
# 	df = df_data[key]
# 	df_fits = pd.read_excel(f"./out/_all_Lognormal/indiv/dt=0.5/{key}_result.xlsx", sheet_name=None, index_col=0)
# 	sheets = list(df_fits.keys())

# 	reader = df['reader']
# 	cp = sns.color_palette("deep", len(reader.condition_names)); max_time = 0
# 	fig1, ax1 = plt.subplots()  
# 	fig2, ax2 = plt.subplots(nrows=3, ncols=1, sharex=True)  # CDFs of the parameters
# 	count = 1
# 	for icnd, cond in enumerate(reader.condition_names):
# 		if cond == 'IL-12' or cond == 'aCD27 + IL-12' or cond == 'aCD28 + IL-12' or cond == 'aCD27 + aCD28 + IL-12':
# 			continue
		
# 		### GET EXPERIMENT INFO
# 		hts = reader.harvested_times[icnd]
# 		mgen = reader.generation_per_condition[icnd]
# 		condition = reader.condition_names[icnd]

# 		### PREPARE DATA
# 		data = df['cgens']['rep'][icnd]; nreps = []
# 		for datum in data:
# 			nreps.append(len(datum))
# 		ax1.errorbar(hts, df['cells']['avg'][icnd], yerr=df['cells']['sem'][icnd], fmt='o', color=cp[icnd], label=f"{''.join(condition.split())}")

# 		if cond == 'N4' or cond == 'aCD27' or cond == 'aCD28':
# 			### SELECT EXCEL SHEETS
# 			select1 = [sheet for sheet in sheets if f"pars_{cond}" == sheet]
# 			select2 = [sheet for sheet in sheets if f"boot_{cond}" == sheet]
			
# 			fit_pars = df_fits[select1[0]]
# 			fit_boots = df_fits[select2[0]]

# 			### GET PARAMETER NAMES AND "VARY" STATES
# 			pars_name = fit_pars.index.to_numpy()
# 			pars_vary = fit_pars.loc[:,'vary']

# 			best_params = lmf.Parameters()
# 			for par in pars_name:
# 				best_params.add(par, value=fit_pars.loc[par, 'best-fit'], vary=pars_vary[par])
# 			best_fit = best_params.valuesdict()
# 			mUns, sUns = best_fit['mUns'], best_fit['sUns']
# 			mDiv0, sDiv0 = best_fit['mDiv0'], best_fit['sDiv0']
# 			mDD, sDD = best_fit['mDD'], best_fit['sDD']
# 			mDie, sDie = best_fit['mDie'], best_fit['sDie']
# 			m, p = best_fit['m'], best_fit['p']

# 			t0, tf, dt = 0, max(hts)+5, 0.5
# 			if tf > max_time: max_time = tf
# 			times = np.linspace(t0, tf, num=int(tf/dt)+1)
# 			gens = np.array([i for i in range(mgen+1)])

# 			### GET CYTON BEST-FIT CURVES
# 			model = Cyton15Model(hts, df['cells']['avg'][icnd][0], mgen, dt, nreps, True)
# 			extrapolate = model.extrapolate(times, best_params)  # get extrapolation for all "times" (discretised) and at harvested timepoints
# 			ext_total_live_cells = extrapolate['ext']['total_live_cells']
# 			ext_cells_per_gen = extrapolate['ext']['cells_gen']
# 			hts_total_live_cells = extrapolate['hts']['total_live_cells']
# 			hts_cells_per_gen = extrapolate['hts']['cells_gen']

# 			# Calculate bootstrap intervals
# 			unst_pdf_curves, unst_cdf_curves = [], []
# 			tdiv0_pdf_curves, tdiv0_cdf_curves = [], []
# 			tdd_pdf_curves, tdd_cdf_curves = [], []
# 			tdie_pdf_curves, tdie_cdf_curves = [], []

# 			b_ext_total_live_cells, b_ext_cells_per_gen = [], []
# 			b_hts_total_live_cells, b_hts_cells_per_gen = [], []

# 			conf = {
# 				'unst_pdf': [], 'unst_cdf': [], 'tdiv0_pdf': [], 'tdiv0_cdf': [], 'tdd_pdf': [], 'tdd_cdf': [], 'tdie_pdf': [], 'tdie_cdf': [],
# 				'ext_total_cohorts': [], 'ext_total_live_cells': [], 'ext_cells_per_gen': [], 'hts_total_live_cells': [], 'hts_cells_per_gen': []
# 			}
# 			tmp_N0 = []  # just to calculate confidence interval for N0, but this is not real parameter! The interval is only from bootstrapping... (And recording this would make easier to plot in the future)
# 			for bsample in fit_boots.drop('algo', axis=1).iterrows():
# 				b_mUns, b_sUns, b_mDiv0, b_sDiv0, b_mDD, b_sDD, b_mDie, b_sDie, b_m, b_p, b_N0 = bsample[1].values
# 				b_params = best_params.copy()
# 				b_params['mUns'].set(value=b_mUns); b_params['sUns'].set(value=b_sUns)
# 				b_params['mDiv0'].set(value=b_mDiv0); b_params['sDiv0'].set(value=b_sDiv0)
# 				b_params['mDD'].set(value=b_mDD); b_params['sDD'].set(value=b_sDD)
# 				b_params['mDie'].set(value=b_mDie); b_params['sDie'].set(value=b_sDie)
# 				b_params['m'].set(value=b_m); b_params['p'].set(value=b_p)

# 				# Calculate PDF and CDF curves for each set of parameter
# 				b_unst_pdf, b_unst_cdf = lognorm_pdf(times, b_mUns, b_sUns), lognorm_cdf(times, b_mUns, b_sUns)
# 				b_tdiv0_pdf, b_tdiv0_cdf = lognorm_pdf(times, b_mDiv0, b_sDiv0), lognorm_cdf(times, b_mDiv0, b_sDiv0)
# 				b_tdd_pdf, b_tdd_cdf = lognorm_pdf(times, b_mDD, b_sDD), lognorm_cdf(times, b_mDD, b_sDD)
# 				b_tdie_pdf, b_tdie_cdf = lognorm_pdf(times, b_mDie, b_sDie), lognorm_cdf(times, b_mDie, b_sDie)

# 				unst_pdf_curves.append(b_unst_pdf); unst_cdf_curves.append(b_unst_cdf)
# 				tdiv0_pdf_curves.append(b_tdiv0_pdf); tdiv0_cdf_curves.append(b_tdiv0_cdf)
# 				tdd_pdf_curves.append(b_tdd_pdf); tdd_cdf_curves.append(b_tdd_cdf)
# 				tdie_pdf_curves.append(b_tdie_pdf); tdie_cdf_curves.append(b_tdie_cdf)

# 				# Calculate model prediction for each set of parameter
# 				b_model = Cyton15Model(hts, b_N0, mgen, dt, nreps, True)
# 				b_extrapolate = b_model.extrapolate(times, b_params)  # get extrapolation for all "times" (discretised) and at harvested timepoints
# 				b_ext_total_live_cells.append(b_extrapolate['ext']['total_live_cells'])
# 				b_ext_total_cohorts = np.sum(np.transpose(b_extrapolate['ext']['cells_gen']) * np.power(2.,-gens), axis=1)
# 				b_ext_cells_per_gen.append(b_extrapolate['ext']['cells_gen'])
# 				b_hts_total_live_cells.append(b_extrapolate['hts']['total_live_cells'])
# 				b_hts_cells_per_gen.append(b_extrapolate['hts']['cells_gen'])

# 				conf['unst_pdf'].append(b_unst_pdf); conf['unst_cdf'].append(b_unst_cdf)
# 				conf['tdiv0_pdf'].append(b_tdiv0_pdf); conf['tdiv0_cdf'].append(b_tdiv0_cdf)
# 				conf['tdd_pdf'].append(b_tdd_pdf); conf['tdd_cdf'].append(b_tdd_cdf)
# 				conf['tdie_pdf'].append(b_tdie_pdf); conf['tdie_cdf'].append(b_tdie_cdf)
# 				conf['ext_total_cohorts'].append(b_ext_total_cohorts)
# 				conf['ext_total_live_cells'].append(b_ext_total_live_cells); conf['ext_cells_per_gen'].append(b_ext_cells_per_gen)
# 				conf['hts_total_live_cells'].append(b_hts_total_live_cells); conf['hts_cells_per_gen'].append(b_hts_cells_per_gen)

# 				tmp_N0.append(b_N0)

# 			# Calculate 95% confidence bands on PDF, CDF and model predictions
# 			for obj in conf:
# 				stack = np.vstack(conf[obj])
# 				conf[obj] = conf_iterval(stack, RGS)
# 			err_mUns, err_sUns = conf_iterval(fit_boots['mUns'], RGS), conf_iterval(fit_boots['sUns'], RGS)
# 			err_mDiv0, err_sDiv0 = conf_iterval(fit_boots['mDiv0'], RGS), conf_iterval(fit_boots['sDiv0'], RGS)
# 			err_mDD, err_sDD = conf_iterval(fit_boots['mDD'], RGS), conf_iterval(fit_boots['sDD'], RGS)
# 			err_mDie, err_sDie = conf_iterval(fit_boots['mDie'], RGS), conf_iterval(fit_boots['sDie'], RGS)
# 			err_m, err_p = conf_iterval(fit_boots['m'], RGS), conf_iterval(fit_boots['p'], RGS)

# 			ax1.set_ylabel("Cell number")
# 			ax1.set_xlabel("Time (hour)")
# 			ax1.plot(times, ext_total_live_cells, color=cp[icnd])
# 			ax1.fill_between(times, conf['ext_total_live_cells'][0], conf['ext_total_live_cells'][1], fc=cp[icnd], ec=None, alpha=0.5)
# 			# ax1.scatter(hts, np.array(df['cells']['avg'][icnd])/df['cells']['avg'][icnd][0], color=cp[icnd], label=f"{condition}")
# 			# ax1.plot(times, ext_total_live_cells/ext_total_live_cells[0], color=cp[icnd])
			
# 			# unst_pdf, unst_cdf = lognorm_pdf(times, mUns, sUns), lognorm_cdf(times, mUns, sUns)
# 			tdiv0_pdf, tdiv0_cdf = lognorm_pdf(times, mDiv0, sDiv0), lognorm_cdf(times, mDiv0, sDiv0)
# 			tdd_pdf, tdd_cdf = lognorm_pdf(times, mDD, sDD), lognorm_cdf(times, mDD, sDD)
# 			tdie_pdf, tdie_cdf = lognorm_pdf(times, mDie, sDie), lognorm_cdf(times, mDie, sDie)
# 			ax2[0].plot(times, tdiv0_cdf, color=cp[icnd], label=f"$\mathcal{{N}}({mDiv0:.2f}\pm_{{{mDiv0-err_mDiv0[0]:.2f}}}^{{{err_mDiv0[1]-mDiv0:.2f}}}, {sDiv0:.2f} \pm_{{{sDiv0-err_sDiv0[0]:.2f}}}^{{{err_sDiv0[1]-sDiv0:.2f}}})$")
# 			ax2[0].fill_between(times, conf['tdiv0_cdf'][0], conf['tdiv0_cdf'][1], fc=cp[icnd], ec=None, alpha=0.5)
# 			ax2[1].plot(times, tdd_cdf, color=cp[icnd], label=f"$\mathcal{{N}}({mDD:.2f}\pm_{{{mDD-err_mDD[0]:.2f}}}^{{{err_mDD[1]-mDD:.2f}}}, {sDD:.2f}\pm_{{{sDD-err_sDD[0]:.2}}}^{{{err_sDD[1]-sDD:.2f}}})$")
# 			ax2[1].fill_between(times, conf['tdd_cdf'][0], conf['tdd_cdf'][1], fc=cp[icnd], ec=None, alpha=0.5)
# 			ax2[2].plot(times, tdie_cdf, color=cp[icnd], label=f"$\mathcal{{N}}({mDie:.2f}\pm_{{{mDie-err_mDie[0]:.2f}}}^{{{err_mDie[1]-mDie:.2f}}}, {sDie:.2f}\pm_{{{sDie-err_sDie[0]:.2f}}}^{{{err_sDie[1]-sDie:.2f}}})$")
# 			ax2[2].fill_between(times, conf['tdie_cdf'][0], conf['tdie_cdf'][1], fc=cp[icnd], ec=None, alpha=0.5)

# 			# ax2[0].annotate(f"[{''.join(condition.split())}]: $m = {m:.2f}\pm_{{{m-err_m[0]:.2f}}}^{{{err_m[1]-m:.2f}}}$", xy=(1, 0.98-count/8), color=cp[icnd], fontsize=14)
# 			# count += 1

# 	### PREDICTION LINE
# 	icnd = 5
# 	lab = 'aCD27+aCD28'

# 	### GET EXPERIMENT INFO
# 	hts = reader.harvested_times[icnd]
# 	mgen = reader.generation_per_condition[icnd]
# 	data = df['cgens']['rep'][icnd]; nreps = []
# 	for datum in data:
# 		nreps.append(len(datum))

# 	pred_pars = df_fits['pars_aCD27 + aCD28']
# 	pars_name = pred_pars.index.to_numpy()
# 	pred_params = lmf.Parameters()
# 	for par in pars_name:
# 		if par == 'mUns': val = 5000
# 		elif par == 'sUns': val = 0.0001
# 		elif par == 'mDiv0': val =  34.22
# 		elif par == 'sDiv0': val = 0.18
# 		elif par == 'mDD': val = 57.75
# 		elif par == 'sDD': val = 0.08
# 		elif par == 'mDie': val = 95.33
# 		elif par == 'sDie': val = 0.39
# 		elif par == 'm': val == 10
# 		elif par == 'p': val = 1
# 		elif par == 'N0': val = 1000
# 		pred_params.add(par, value=val)

# 	## Calculate bootstrap intervals
# 	unst_pdf_curves, unst_cdf_curves = [], []
# 	tdiv0_pdf_curves, tdiv0_cdf_curves = [], []
# 	tdd_pdf_curves, tdd_cdf_curves = [], []
# 	tdie_pdf_curves, tdie_cdf_curves = [], []

# 	pb_ext_total_live_cells, pb_ext_cells_per_gen = [], []
# 	pb_hts_total_live_cells, pb_hts_cells_per_gen = [], []
# 	conf = {
# 		'unst_pdf': [], 'unst_cdf': [], 'tdiv0_pdf': [], 'tdiv0_cdf': [], 'tdd_pdf': [], 'tdd_cdf': [], 'tdie_pdf': [], 'tdie_cdf': [],
# 		'ext_total_cohorts': [], 'ext_total_live_cells': [], 'ext_cells_per_gen': [], 'hts_total_live_cells': [], 'hts_cells_per_gen': []
# 	}
# 	pred_boots = df_fits['boot_aCD27 + aCD28'].dropna(axis=0)
# 	for pbsample in pred_boots.iterrows():
# 		pb_mUns, pb_sUns, pb_mDiv0, pb_sDiv0, pb_mDD, pb_sDD, pb_mDie, pb_sDie, pb_m, pb_p, pb_N0, _ = pbsample[1].values
# 		pb_params = pred_params.copy()
# 		pb_params['mUns'].set(value=pb_mUns); pb_params['sUns'].set(value=pb_sUns)
# 		pb_params['mDiv0'].set(value=pb_mDiv0); pb_params['sDiv0'].set(value=pb_sDiv0)
# 		pb_params['mDD'].set(value=pb_mDD); pb_params['sDD'].set(value=pb_sDD)
# 		pb_params['mDie'].set(value=pb_mDie); pb_params['sDie'].set(value=pb_sDie)
# 		pb_params['m'].set(value=pb_m); pb_params['p'].set(value=pb_p)

# 		# Calculate PDF and CDF curves for each set of parameter
# 		pb_unst_pdf, pb_unst_cdf = lognorm_pdf(times, pb_mUns, pb_sUns), lognorm_cdf(times, pb_mUns, pb_sUns)
# 		pb_tdiv0_pdf, pb_tdiv0_cdf = lognorm_pdf(times, pb_mDiv0, pb_sDiv0), lognorm_cdf(times, pb_mDiv0, pb_sDiv0)
# 		pb_tdd_pdf, pb_tdd_cdf = lognorm_pdf(times, pb_mDD, pb_sDD), lognorm_cdf(times, pb_mDD, pb_sDD)
# 		pb_tdie_pdf, pb_tdie_cdf = lognorm_pdf(times, pb_mDie, pb_sDie), lognorm_cdf(times, pb_mDie, pb_sDie)

# 		unst_pdf_curves.append(pb_unst_pdf); unst_cdf_curves.append(pb_unst_cdf)
# 		tdiv0_pdf_curves.append(pb_tdiv0_pdf); tdiv0_cdf_curves.append(pb_tdiv0_cdf)
# 		tdd_pdf_curves.append(pb_tdd_pdf); tdd_cdf_curves.append(pb_tdd_cdf)
# 		tdie_pdf_curves.append(pb_tdie_pdf); tdie_cdf_curves.append(pb_tdie_cdf)

# 		# Calculate model prediction for each set of parameter
# 		pb_model = Cyton15Model(hts, pb_N0, mgen, dt, nreps, True)
# 		pb_extrapolate = pb_model.extrapolate(times, pb_params)  # get extrapolation for all "times" (discretised) and at harvested timepoints
# 		pb_ext_total_live_cells.append(pb_extrapolate['ext']['total_live_cells'])
# 		pb_ext_total_cohorts = np.sum(np.transpose(pb_extrapolate['ext']['cells_gen']) * np.power(2.,-gens), axis=1)
# 		pb_ext_cells_per_gen.append(pb_extrapolate['ext']['cells_gen'])
# 		pb_hts_total_live_cells.append(pb_extrapolate['hts']['total_live_cells'])
# 		pb_hts_cells_per_gen.append(pb_extrapolate['hts']['cells_gen'])

# 		conf['unst_pdf'].append(pb_unst_pdf); conf['unst_cdf'].append(pb_unst_cdf)
# 		conf['tdiv0_pdf'].append(pb_tdiv0_pdf); conf['tdiv0_cdf'].append(pb_tdiv0_cdf)
# 		conf['tdd_pdf'].append(pb_tdd_pdf); conf['tdd_cdf'].append(pb_tdd_cdf)
# 		conf['tdie_pdf'].append(pb_tdie_pdf); conf['tdie_cdf'].append(pb_tdie_cdf)
# 		conf['ext_total_cohorts'].append(pb_ext_total_cohorts)
# 		conf['ext_total_live_cells'].append(pb_ext_total_live_cells); conf['ext_cells_per_gen'].append(pb_ext_cells_per_gen)
# 		conf['hts_total_live_cells'].append(pb_hts_total_live_cells); conf['hts_cells_per_gen'].append(pb_hts_cells_per_gen)
# 	for obj in conf:
# 		stack = np.vstack(conf[obj])
# 		conf[obj] = conf_iterval(stack, RGS)
# 	err_mUns, err_sUns = conf_iterval(pred_boots['mUns'], RGS), conf_iterval(pred_boots['sUns'], RGS)
# 	err_mDiv0, err_sDiv0 = conf_iterval(pred_boots['mDiv0'], RGS), conf_iterval(pred_boots['sDiv0'], RGS)
# 	err_mDD, err_sDD = conf_iterval(pred_boots['mDD'], RGS), conf_iterval(pred_boots['sDD'], RGS)
# 	err_mDie, err_sDie = conf_iterval(pred_boots['mDie'], RGS), conf_iterval(pred_boots['sDie'], RGS)
# 	err_m, err_p = conf_iterval(pred_boots['m'], RGS), conf_iterval(pred_boots['p'], RGS)

# 	unst_pdf, unst_cdf = lognorm_pdf(times, pred_params['mUns'], pred_params['sUns']), lognorm_cdf(times, pred_params['mUns'], pred_params['sUns'])
# 	tdiv0_pdf, tdiv0_cdf = lognorm_pdf(times, pred_params['mDiv0'], pred_params['sDiv0']), lognorm_cdf(times, pred_params['mDiv0'], pred_params['sDiv0'])
# 	tdd_pdf, tdd_cdf = lognorm_pdf(times, pred_params['mDD'], pred_params['sDD']), lognorm_cdf(times, pred_params['mDD'], pred_params['sDD'])
# 	tdie_pdf, tdie_cdf = lognorm_pdf(times, pred_params['mDie'], pred_params['sDie']), lognorm_cdf(times, pred_params['mDie'], pred_params['sDie'])
# 	m = pred_pars.loc['m', 'best-fit']
# 	ax2[0].annotate(f"$m = {m:.2f}\pm_{{{m-err_m[0]:.2f}}}^{{{err_m[1]-m:.2f}}}$", xy=(1, 0.85), color='k', fontsize=16)
# 	ax2[0].plot(times, tdiv0_cdf, '--', color=cp[icnd], 
# 				label=f"$\mathcal{{LN}}({pred_params['mDiv0'].value:.2f}\pm_{{{pred_params['mDiv0'].value-err_mDiv0[0]:.2f}}}^{{{err_mDiv0[1]-pred_params['mDiv0'].value:.2f}}}, {pred_params['sDiv0'].value:.2f} \pm_{{{pred_params['sDiv0'].value-err_sDiv0[0]:.2f}}}^{{{err_sDiv0[1]-pred_params['sDiv0'].value:.2f}}})$")
# 	ax2[0].fill_between(times, conf['tdiv0_cdf'][0], conf['tdiv0_cdf'][1], fc=cp[icnd], ec=None, alpha=0.5)
# 	ax2[1].plot(times, tdd_cdf, '--', color=cp[icnd], 
# 				label=f"$\mathcal{{LN}}({pred_params['mDD'].value:.2f}\pm_{{{pred_params['mDD'].value-err_mDD[0]:.2f}}}^{{{err_mDD[1]-pred_params['mDD'].value:.2f}}}, {pred_params['sDD'].value:.2f}\pm_{{{pred_params['sDD'].value-err_sDD[0]:.2f}}}^{{{err_sDD[1]-pred_params['sDD'].value:.2f}}})$")
# 	ax2[1].fill_between(times, conf['tdd_cdf'][0], conf['tdd_cdf'][1], fc=cp[icnd], ec=None, alpha=0.5)
# 	ax2[2].plot(times, tdie_cdf, '--', color=cp[icnd], 
# 				label=f"$\mathcal{{LN}}({pred_params['mDie'].value:.2f}\pm_{{{pred_params['mDie'].value-err_mDie[0]:.2f}}}^{{{err_mDie[1]-pred_params['mDie'].value:.2f}}}, {pred_params['sDie'].value:.2f}\pm_{{{pred_params['sDie'].value-err_sDie[0]:.2f}}}^{{{err_sDie[1]-pred_params['sDie'].value:.2f}}})$")
# 	ax2[2].fill_between(times, conf['tdie_cdf'][0], conf['tdie_cdf'][1], fc=cp[icnd], ec=None, alpha=0.5)

# 	pred_model = Cyton15Model(hts, df['cells']['avg'][icnd][0], mgen, dt, nreps, True)
# 	extrapolate = pred_model.extrapolate(times, pred_params)
# 	ax1.plot(times, extrapolate['ext']['total_live_cells'], '--', color=cp[icnd], label='Predicted')
# 	ax1.fill_between(times, conf['ext_total_live_cells'][0], conf['ext_total_live_cells'][1], fc=cp[icnd], ec=None, alpha=0.5)

# 	handles, labels = ax1.get_legend_handles_labels()
# 	handles = handles[1:] + handles[0:1]
# 	handles.reverse()
# 	labels = labels[1:] + labels[0:1]
# 	labels.reverse()
# 	ax1.legend(labels=labels, handles=handles, frameon=True, fontsize=20)
# 	ax1.set_xlim(left=0, right=max_time)
# 	# ax1.set_yscale('log')
# 	# ax1.set_ylim(bottom=1E3)
# 	ax1.set_ylim(bottom=0)
# 	ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# 	ax1.yaxis.major.formatter._useMathText = True
# 	ax2_titles = ["Time to first division ($T_{div}^0$)", "Time to division destiny ($T_{dd}$)", "Time to death ($T_{die}$)"]
# 	for ix, axis in enumerate(ax2):
# 		axis.set_title(ax2_titles[ix], x=0.01, ha='left')
# 		axis.set_ylabel("CDF")
# 		axis.set_ylim(bottom=0, top=1)
# 		axis.set_xlim(left=0, right=max_time)
# 		handles, labels = axis.get_legend_handles_labels()
# 		handles.reverse()
# 		labels.reverse()
# 		axis.legend(ncol=1, handles=handles, labels=labels, frameon=False, fontsize=16)
# 	ax2[2].set_xlabel("Time (hour)")
# 	fig1.tight_layout(rect=(0, 0, 1, 1))
# 	fig2.tight_layout(rect=(0, 0, 1, 1))
# 	fig2.subplots_adjust(wspace=0, hspace=0.2)

# 	# fig1.savefig(f"./out/{key}_cells.pdf", dpi=300)
# 	# fig2.savefig(f"./out/{key}_params.pdf", dpi=300)

# 	plt.show()