"""
Last edit: 16-May-2021

ABM simulation code to recreate Marchingo et al. 2014 Fig.3 results. In particular, the linear sum of MDN.
Two ways to compute the predicted MDN for aCD27 + aCD28:
(i) Sum increase of MDNs from individual components as described in JM-Science 2014 article.
(ii) Simulate the trees with summed times (i.e. Tdiv0, Tdd and Tdie) as described in the main article.
[Output] FigS7 in the Supplementary Material
"""
import os, time, datetime, tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as sps
import matplotlib as mpl; mpl.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import multiprocessing as mp
from src.parse import parse_data
from src.cell import ABM

rc = {
	'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
	'xtick.labelsize': 14, 'ytick.labelsize': 14,
	'figure.figsize': (9, 11),
	'axes.grid': True, 'grid.linestyle': ':', 'axes.grid.which': 'both',
	'axes.axisbelow': True, 'axes.titlepad': 0,
	'axes.spines.top': False, 'axes.spines.right': False,
	'axes.spines.left': True, 'axes.spines.bottom': True,
	'ytick.left': True, 'xtick.bottom': True,
	'lines.markersize': 5.5, 'lines.linewidth': 1.5,
	'errorbar.capsize': 2.5
}
sns.set(context='paper', style='white', rc=rc)

""" DATA DESCRIPTION
EX127: Fig.3 of Marchingo et al. Science 2014
	(1) N4, (2) aCD27, (3) aCD28, (4) IL12, (5) aCD27+IL12, (6) aCD27+aCD28, (7) aCD28+IL12, (8) aCD27+aCD28+IL12
EX130b: repeat of EX127
"""
PATH_TO_DATA = './data'
DATA = ["EX127.xlsx", "EX130b.xlsx"]
KEYS = [os.path.splitext(os.path.basename(data_key))[0] for data_key in DATA]

RGS = 95
DT = 0.5
df_data = parse_data(PATH_TO_DATA, DATA)

def run_abm(inputs):
	key, cond, pars, rgs, hts, dt, max_gen, n0, result = inputs

	pos = mp.current_process()._identity[0]-1  # For progress bar
	abm = ABM(rgs=rgs, t0=0, tf=max(hts), dt=dt, max_gen=max_gen, n0=n0)
	abm.run(pos=pos, name=f"{key} {cond}", pars=pars, n_sims=1000)

	# Calculate key statistics
	abm.total_cohort(hts=hts)
	abm.mdn(hts=hts)

	# Construct buffer dictionary
	tmp = {}
	tmp[cond] = abm

	# Update shared dictionary object
	foo = result[key]
	foo.update(tmp)
	result[key] = foo

if __name__ == "__main__":
	start = time.time()

	inputs = []
	manager = mp.Manager()
	store_abm = manager.dict({'{}'.format(key): {} for key in KEYS}) 
	store_data = {
		f'{key}': { 
			f'{cond}': 
				{	
					'hts': None, 'mgen': None, 'nreps': None,
					'mdn': None, 'mdn_sem': None, 
					'total_cohort_norm': None, 'total_cohort_norm_sem': None
				} 
				for cond in df_data[key]['reader'].condition_names 
			} for key in KEYS }

	# Prepare the data and simulation inputs
	for key in KEYS:
		df = df_data[key]
		df_fits = pd.read_excel(f"./out/_normal/joint/Fig6/fitResults/{key}_result.xlsx", sheet_name=None, index_col=0)
		sheets = list(df_fits.keys())
		reader = df['reader']
		for icnd, cond in enumerate(reader.condition_names):
			if cond == 'IL-12' or cond == 'aCD27 + IL-12' or cond == 'aCD28 + IL-12' or cond == 'aCD27 + aCD28 + IL-12':
				continue

			### SELECT EXCEL SHEETS
			if cond == 'aCD27 + aCD28':
				select1 = [sheet for sheet in sheets if f"pars_pred_aCD27_aCD28" == sheet]
				select2 = [sheet for sheet in sheets if f"boot_pred_aCD27_aCD28" == sheet]
			else:
				select1 = [sheet for sheet in sheets if f"pars_{cond}" == sheet]
				select2 = [sheet for sheet in sheets if f"boot_{cond}" == sheet]

			fit_pars = df_fits[select1[0]]
			fit_boots = df_fits[select2[0]]

			### GET PARAMETER NAMES AND "VARY" STATES
			pars_name = fit_pars.index.to_numpy()

			### GET EXPERIMENT INFO
			hts = reader.harvested_times[icnd]
			mgen = reader.generation_per_condition[icnd]

			### PREPARE DATA
			data = df['cgens']['rep'][icnd]; nreps = []
			for datum in data:
				nreps.append(len(datum))

			pars = {f'{par}': fit_pars.loc[par, 'best-fit'] for par in pars_name}

			### PLOT TOTAL COHORTS (+DATA)
			tps, total_cohorts = [], []
			total_cohorts_norm, total_cohorts_norm_sem = [], []
			for itpt, ht in enumerate(hts):
				tmp = []
				for irep in range(nreps[itpt]):
					tps.append(ht)
					sum_cohort = np.sum(df['cohorts_gens']['rep'][icnd][itpt][irep])
					total_cohorts.append(sum_cohort)
					if not itpt:
						sum_cohort0 = sum_cohort
					tmp.append(sum_cohort/sum_cohort0 * 100)
				total_cohorts_norm.append(np.mean(tmp))
				total_cohorts_norm_sem.append(sps.sem(tmp))

			### PLOT MEAN DIVISION NUMBER (+DATA)
			tps, mdn, mdn_avg, mdn_sem = [], [], [], []
			for itpt, ht in enumerate(hts):
				tmp = []
				for irep in range(nreps[itpt]):
					tps.append(ht)

					cohort = np.array(df['cohorts_gens']['rep'][icnd][itpt][irep])
					sum_cohort = np.sum(cohort)

					tmp_mdn = np.sum(cohort * np.array([igen for igen in range(mgen+1)]) / sum_cohort)
					mdn.append(tmp_mdn)

					tmp.append(tmp_mdn)
				mdn_avg.append(np.mean(tmp))
				mdn_sem.append(sps.sem(tmp))
			
			## Store abm
			store_data[key][f'{cond}']['hts'] = hts
			store_data[key][f'{cond}']['mgen'] = mgen
			store_data[key][f'{cond}']['nreps'] = nreps
			store_data[key][f'{cond}']['total_cohort_norm'] = total_cohorts_norm
			store_data[key][f'{cond}']['total_cohort_norm_sem'] = total_cohorts_norm_sem
			store_data[key][f'{cond}']['mdn'] = mdn_avg
			store_data[key][f'{cond}']['mdn_sem'] = mdn_sem

			if cond == 'aCD27 + aCD28':
				inputs.append((key, cond, pars, RGS, hts, DT, mgen, int(df['cells']['avg'][icnd][0]), store_abm))
			else:
				inputs.append((key, cond, pars, RGS, hts, DT, mgen, int(pars['N0']), store_abm))

	# Run the simulation in parallel
	tqdm.tqdm.set_lock(mp.RLock())  # for managing output contention
	p = mp.Pool(initializer=tqdm.tqdm.set_lock, initargs=(tqdm.tqdm.get_lock(),))
	with tqdm.tqdm(total=len(inputs), desc="Total", position=0) as pbar:
		for i, _ in enumerate(p.imap_unordered(run_abm, inputs)):
			pbar.update()
	p.close()
	p.join()

	# Plot results
	for key in KEYS:
		df = df_data[key]
		reader = df['reader']
		for icnd, cond in enumerate(reader.condition_names):
			if cond == 'IL-12' or cond == 'aCD27 + IL-12' or cond == 'aCD28 + IL-12' or cond == 'aCD27 + aCD28 + IL-12':
				continue

			hts, mgen, nreps = store_data[key][cond]['hts'], store_data[key][cond]['mgen'], store_data[key][cond]['nreps']
			abm = store_abm[key][cond]

			### 95% CONFIDENCE BADNS
			alpha = (1. - RGS/100.)/2

			quants = {f'gen{igen}': [] for igen in range(abm.max_gen+1)}
			quants['total'] = []
			for n, df_abm in enumerate(abm.dfs):
				for key_quant, quant in quants.items():
					quants[key_quant].append(df_abm[key_quant].to_list())
					if n == len(abm.dfs)-1:
						quants[key_quant] = np.array(quants[key_quant])  # stack the lists vertically

			conf_bands = {
				'avg': { f'gen{igen}': [] for igen in range(abm.max_gen+1) },
				'low': { f'gen{igen}': [] for igen in range(abm.max_gen+1) },
				'upp': { f'gen{igen}': [] for igen in range(abm.max_gen+1) }
			}
			conf_bands['low']['total'], conf_bands['upp']['total'] = [], []
			for key_conf, quant in quants.items():
				avg = np.mean(quants[key_conf], axis=0)
				low, upp = np.quantile(quants[key_conf], [alpha, alpha + RGS/100], interpolation='nearest', axis=0)
				conf_bands['avg'][key_conf] = avg
				conf_bands['low'][key_conf] = low
				conf_bands['upp'][key_conf] = upp
			
			### PLOT CELL NUMBERS PER GENERATION vs TIME (+DATA)
			cp = sns.hls_palette(mgen+1, l=0.4, s=0.5)
			fig, ax = plt.subplots(nrows=3, sharex=True)
			# fig.suptitle(f"{cond}")
			for igen in range(mgen+1):
				ax[0].plot(abm.times, conf_bands['avg'][f'gen{igen}'], c=cp[igen])
				ax[0].fill_between(abm.times, conf_bands['low'][f'gen{igen}'], conf_bands['upp'][f'gen{igen}'], color=cp[igen], alpha=0.3)
				ax[0].errorbar(hts, np.transpose(df['cgens']['avg'][icnd])[igen], yerr=np.transpose(df['cgens']['sem'][icnd])[igen], c='k', mfc=cp[igen], fmt='o', ms=5, label=f"Gen {igen}")
			ax[0].set_ylabel("Cell number")
			ax[0].set_title(f"[{cond}] Live cell numbers")
			ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
			ax[0].yaxis.major.formatter._useMathText = True
			ax[0].legend(ncol=1, fontsize=14)

			### PLOT TOTAL COHORTS (+DATA)
			tps, total_cohorts = [], []
			total_cohorts_norm, total_cohorts_norm_sem = [], []
			for itpt, ht in enumerate(hts):
				tmp = []
				for irep in range(nreps[itpt]):
					tps.append(ht)
					sum_cohort = np.sum(df['cohorts_gens']['rep'][icnd][itpt][irep])
					total_cohorts.append(sum_cohort)
					if not itpt:
						sum_cohort0 = sum_cohort
					tmp.append(sum_cohort/sum_cohort0 * 100)
				total_cohorts_norm.append(np.mean(tmp))
				total_cohorts_norm_sem.append(sps.sem(tmp))
			ax[1].plot(tps, total_cohorts, 'ro', label='data')
			ax[1].plot(abm.times, abm.total_cohort_times['avg'], c='k')
			ax[1].fill_between(abm.times, abm.total_cohort_times['low'], abm.total_cohort_times['upp'], color='k', alpha=0.3)
			ax[1].set_ylabel("Cohort number")
			ax[1].set_title("Total cohort number")
			ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
			ax[1].yaxis.major.formatter._useMathText = True
			ax[1].legend()

			### PLOT MEAN DIVISION NUMBER (+DATA)
			tps, mdn, mdn_avg, mdn_sem = [], [], [], []
			for itpt, ht in enumerate(hts):
				tmp = []
				for irep in range(nreps[itpt]):
					tps.append(ht)

					cohort = np.array(df['cohorts_gens']['rep'][icnd][itpt][irep])
					sum_cohort = np.sum(cohort)

					tmp_mdn = np.sum(cohort * np.array([igen for igen in range(mgen+1)]) / sum_cohort)
					mdn.append(tmp_mdn)

					tmp.append(tmp_mdn)
				mdn_avg.append(np.mean(tmp))
				mdn_sem.append(sps.sem(tmp))
			ax[2].plot(tps, mdn, 'ro', label='data')
			ax[2].plot(abm.times, abm.mdn_times['avg'], c='k')
			ax[2].fill_between(abm.times, abm.mdn_times['low'], abm.mdn_times['upp'], color='k', alpha=0.3)
			ax[2].set_ylabel("Mean div. number")
			ax[2].set_xlabel("Time (hour)")
			ax[2].set_title("Mean division number")
			ax[2].set_xlim(left=min(abm.times), right=max(abm.times)+2)
			# ax[2].yaxis.set_major_locator(MaxNLocator(integer=True))
			# ax[2].tick_params(axis='y', which='minor', bottom=False)

			for axis in ax:
				axis.set_ylim(bottom=0)
			fig.tight_layout(rect=(0, 0, 1, 1))
			fig.subplots_adjust(wspace=0, hspace=0.13)
			fig.savefig(f"./out/FigS7-Simulation/{key}_{cond}.pdf", dpi=300)

		cbasis = sns.color_palette("deep", len(reader.condition_names))
		cp = {
			'N4': cbasis[0], 
			'aCD27': cbasis[1], 
			'aCD28': cbasis[2], 
			'aCD27 + aCD28': cbasis[5]
		}
		fig2, ax2 = plt.subplots(nrows=3, sharex=True)
		for cond, abm in store_abm[key].items():
			if cond == 'N4': continue
			elif cond == 'aCD27': i = 0
			elif cond == 'aCD28': i = 1
			elif cond == 'aCD27 + aCD28': i = 2
			
			if cond == 'aCD27' or cond == 'aCD28' or cond == 'aCD27 + aCD28':
				y = 10
				# max_N4_mdn = max(store_data[key]['N4']['mdn'])  # Max MDN from data
				max_N4_mdn = max(store_abm[key]['N4'].mdn_times['avg'])
				lowN4_mdn = max(store_abm[key]['N4'].mdn_times['low'])
				uppN4_mdn = max(store_abm[key]['N4'].mdn_times['upp'])
				ax2[i].axvline(max_N4_mdn, c=cp['N4'], ls=':')
				ax2[i].arrow(x=0, y=y, dx=max_N4_mdn, dy=0, color=cp['N4'], head_width=5, head_length=0.05, lw=0.5, ls="-", length_includes_head=True)
				ax2[i].arrow(x=max_N4_mdn, y=y, dx=-max_N4_mdn, dy=0, color=cp['N4'], head_width=5, head_length=0.05, lw=0.5, ls="-", length_includes_head=True)
				if cond == 'aCD27':
					ax2[i].text(x=max_N4_mdn/3, y=y+2.1, s=f"+{max_N4_mdn:.2f}$\pm_{{{max_N4_mdn - lowN4_mdn:.2f}}}^{{{uppN4_mdn - max_N4_mdn:.2f}}}$", color=cp['N4'], ha='left')
					ax2[i].errorbar(store_data[key]['N4']['mdn'], store_data[key]['N4']['total_cohort_norm'], 
									xerr=store_data[key]['N4']['mdn_sem'], yerr=store_data[key]['N4']['total_cohort_norm_sem'], 
									fmt='o', c='k', mfc=cp['N4'], label='N4')
					ax2[i].plot(store_abm[key]['N4'].mdn_times['avg'], 
								np.array(store_abm[key]['N4'].total_cohort_times['avg'])/store_abm[key]['N4'].n0*100, c=cp['N4'])
					ax2[i].fill(np.append(store_abm[key]['N4'].mdn_times['low'], store_abm[key]['N4'].mdn_times['upp'][::-1]),
								np.append(np.array(store_abm[key]['N4'].total_cohort_times['low'])/store_abm[key]['N4'].n0*100, 
								np.array(store_abm[key]['N4'].total_cohort_times['upp'])[::-1]/store_abm[key]['N4'].n0*100), color=cp['N4'], alpha=0.3)

				mdn = store_data[key][cond]['mdn']
				mdn_sem = store_data[key][cond]['mdn_sem']
				total_cohort_norm = store_data[key][cond]['total_cohort_norm']
				total_cohort_norm_sem = store_data[key][cond]['total_cohort_norm_sem']
				if cond == 'aCD27':
					ax2[i].errorbar(mdn, total_cohort_norm, 
									xerr=mdn_sem, yerr=total_cohort_norm_sem, 
									fmt='o', c='k', mfc=cp[cond], label=r"$\alpha$CD27")
				elif cond == 'aCD28':
					ax2[i].errorbar(mdn, total_cohort_norm, 
									xerr=mdn_sem, yerr=total_cohort_norm_sem, 
									fmt='o', c='k', mfc=cp[cond], label=r"$\alpha$CD28")
				elif cond == 'aCD27 + aCD28':
					ax2[i].errorbar(mdn, total_cohort_norm, 
									xerr=mdn_sem, yerr=total_cohort_norm_sem, 
									fmt='o', c='k', mfc=cp[cond], label=r"$\alpha$CD27+$\alpha$CD28")

				if cond == 'aCD27' or cond == 'aCD28':
					ax2[i].plot(abm.mdn_times['avg'], np.array(abm.total_cohort_times['avg'])/abm.n0*100, c=cp[cond])
					ax2[i].fill(np.append(abm.mdn_times['low'], abm.mdn_times['upp'][::-1]),
								np.append(np.array(abm.total_cohort_times['low'])/abm.n0*100, 
								np.array(abm.total_cohort_times['upp'])[::-1]/abm.n0*100), color=cp[cond], alpha=0.3)
					
					x = max(abm.mdn_times['avg']); dx = max(abm.mdn_times['avg']) - max_N4_mdn
					lowx = max(abm.mdn_times['low'])
					uppx = max(abm.mdn_times['upp'])
					ax2[i].axvline(x, c=cp[cond], ls=':')  # Plot max of data
					ax2[i].text(x=max_N4_mdn + dx/5, y=y+2.1, s=f"+{dx:.2f}$\pm_{{{x-lowx:.2f}}}^{{{uppx-x:.2f}}}$", color=cp[cond], ha='left')
					ax2[i].arrow(x=max_N4_mdn, y=y, dx=dx, dy=0, color=cp[cond], head_width=5, head_length=0.05, lw=0.5, ls="-", length_includes_head=True)
					ax2[i].arrow(x=x, y=y, dx=-dx, dy=0, color=cp[cond], head_width=5, head_length=0.05, lw=0.5, ls="-", length_includes_head=True)
				elif cond == 'aCD27 + aCD28':
					simul_avgMDN = max(abm.mdn_times['avg'])
					simul_lowMDN = max(abm.mdn_times['low'])
					simul_uppMDN = max(abm.mdn_times['upp'])
					ax2[i].plot(abm.mdn_times['avg'], np.array(abm.total_cohort_times['avg'])/abm.n0*100, c=cp[cond], 
								label=f"By simulation: {simul_avgMDN:.2f}$\pm_{{{simul_avgMDN - simul_lowMDN:.2f}}}^{{{simul_uppMDN - simul_avgMDN:.2f}}}$")
					ax2[i].fill(np.append(abm.mdn_times['low'], abm.mdn_times['upp'][::-1]),
								np.append(np.array(abm.total_cohort_times['low'])/abm.n0*100, 
								np.array(abm.total_cohort_times['upp'])[::-1]/abm.n0*100), color=cp[cond], alpha=0.3)
						
					# dx_aCD27 = max(store_data[key][cond]['mdn']) - max_N4_mdn
					# dx_aCD28 = max(store_data[key][cond]['mdn']) - max_N4_mdn
					# pred = max(store_data[key][cond]['mdn']) + max(store_data[key][cond]['mdn']) - max_N4_mdn
					dx_aCD27 = max(store_abm[key]['aCD27'].mdn_times['avg']) - max_N4_mdn
					dx_lowaCD27 = max(store_abm[key]['aCD27'].mdn_times['low']) - max(store_abm[key]['N4'].mdn_times['low'])
					dx_uppaCD27 = max(store_abm[key]['aCD27'].mdn_times['upp']) - max(store_abm[key]['N4'].mdn_times['upp'])

					dx_aCD28 = max(store_abm[key]['aCD28'].mdn_times['avg']) - max_N4_mdn
					dx_lowaCD28 = max(store_abm[key]['aCD28'].mdn_times['low']) - max(store_abm[key]['N4'].mdn_times['low'])
					dx_uppaCD28 = max(store_abm[key]['aCD28'].mdn_times['upp']) - max(store_abm[key]['N4'].mdn_times['upp'])
					
					pred = max_N4_mdn + dx_aCD27 + dx_aCD28
					lowPred = max(store_abm[key]['N4'].mdn_times['low']) + dx_lowaCD27 + dx_lowaCD28
					uppPred = max(store_abm[key]['N4'].mdn_times['upp']) + dx_uppaCD27 + dx_uppaCD28
					ax2[i].axvline(pred, c=cp[cond], ls='--', label=f"By addition: {pred:.2f}$\pm_{{{pred - lowPred:.2f}}}^{{{uppPred - pred:.2f}}}$")
					ax2[i].axvspan(lowPred, uppPred, color=cp[cond], alpha=0.3)

					# ax2[i].text(x=max_N4_mdn + dx_aCD27/3, y=y+2.1, s=f"+{dx_aCD27:.2f}", color=cp['aCD27'], ha='left')
					ax2[i].arrow(x=max_N4_mdn, y=y, dx=dx_aCD27, dy=0, color=cp['aCD27'], head_width=5, head_length=0.05, lw=0.5, ls="-", length_includes_head=True)
					ax2[i].arrow(x=max_N4_mdn+dx_aCD27, y=y, dx=-dx_aCD27, dy=0, color=cp['aCD27'], head_width=5, head_length=0.05, lw=0.5, ls="-", length_includes_head=True)

					# ax2[i].text(x=max_N4_mdn + dx_aCD27 + dx_aCD28/3, y=y+2.1, s=f"+{dx_aCD28:.2f}", color=cp['aCD28'], ha='left')
					ax2[i].arrow(x=max_N4_mdn+dx_aCD27, y=y, dx=dx_aCD28, dy=0, color=cp['aCD28'], head_width=5, head_length=0.05, lw=0.5, length_includes_head=True)
					ax2[i].arrow(x=max_N4_mdn+dx_aCD27+dx_aCD28, y=y, dx=-dx_aCD28, dy=0, color=cp['aCD28'], head_width=5, head_length=0.05, lw=0.5, length_includes_head=True)

		# fig2.text(0.02, 0.5, "% Cohort number", ha='center', va='center', rotation=90, fontsize=rc['axes.labelsize'])
		for i, axis in enumerate(ax2):
			axis.set_ylabel("% Cohort number")
			axis.set_ylim(bottom=0, top=110)
			axis.set_xlim(left=0)
			if i < 2:
				axis.legend(fontsize=14)
			else:
				handles, labels = axis.get_legend_handles_labels()
				# handles = handles[1:] + handles[0:1]
				# handles.reverse()
				# labels = labels[1:] + labels[0:1]
				# labels.reverse()
				axis.legend(handles=handles, labels=labels, fontsize=14)
		handles, labels = ax2[0].get_legend_handles_labels()
		ax2[0].legend(handles=[handles[0], handles[1]], labels=[labels[0], labels[1]], fontsize=14)
		ax2[2].set_xlabel("Mean division number (MDN)")
		# ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
		# ax2.tick_params(axis='x', which='minor', bottom=False)
		fig2.tight_layout(rect=(0.01, 0, 1, 1))
		# fig2.subplots_adjust(wspace=0, hspace=0.05)
		fig2.subplots_adjust(wspace=0, hspace=0)
		fig2.savefig(f"./out/FigS7-Simulation/{key}_mdn.pdf", dpi=300)
		# plt.show()
	
	end = time.time()
	hours, rem = divmod(end-start, 3600)
	minutes, seconds = divmod(rem, 60)
	now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
	print(f"> DONE ! {now}")
	print("> Elapsed Time = {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
