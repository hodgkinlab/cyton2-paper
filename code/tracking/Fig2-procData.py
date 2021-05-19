"""
Last edit: 16-May-2021

Process filming data for B (B-exp1, B-exp2) and CD8+ T cells (IL-2, T-exp1, T-exp2)
[Output] Fig2 in the main article; FigS1, FigS2, FigS3 in the Supplementary Material
"""
import sys, os, itertools
os.environ['MKL_NUM_THREADS'] = "1"  # pymc3 MVN uses numpy to calculate correlation coefficient, which by default uses all available cores
os.environ['OMP_NUM_THREADS'] = "1"  # both are required to limit number of threads used by numpy
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 999999)
pd.set_option('display.max_columns', 999999)
pd.set_option('display.expand_frame_repr', False)
import seaborn as sns
import matplotlib as mpl; mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy as sp
import scipy.stats as sps
import arviz as az
import pymc3 as pm
import theano as TT
import theano.tensor as T
from _func import parse_data, filter_data, rank_mean_fam, save_dataframes, save_cc_times

## Check library versions
try:
	assert(pm.__version__=='3.9.3')
	assert(TT.__version__=='1.0.5')
	assert(az.__version__=='0.11.0')
	assert(sns.__version__=='0.10.1')
except AssertionError as ae:
	print("[VersionError] Please check if the following versions of PyMC3, Theano, Arviz, seaborn libraries are installed:")
	print("pymc3==3.9.3")
	print("Theano==1.0.5")
	print("arviz==0.11.0")
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

bcpg_cpg3 = {0.: 'CpG'}
bcpg_cpg4 = {0.: 'CpG'}
til2_20131218 = {1.: '1U IL-2', 2.: '3U IL-2', 3.: '10U IL-2'}
til2_20140121 = {1.: '1U IL-2', 2.: '10U IL-2', 3.: '3U IL-2'}
tmisc_20140211 = {1.: 'N4+CD28+IL-2', 2.: 'N4+CD28', 3.: 'N4', 4.: 'N4+IL-2'}
tmisc_20140325 = {1.: 'N4+IL-12', 2.: 'N4+CD28+IL-12', 3.: 'N4+CD28', 4.: 'N4'}
artf_aggre_1U = {1.: '1U IL-2'}
artf_aggre_3U = {2.: '3U IL-2'}
artf_aggre_10U = {3.: '10U IL-2'}
cond_labs = {
	'b_cpg/cpg3': bcpg_cpg3, 
	'b_cpg/cpg4': bcpg_cpg4, 
	't_il2/20140121': til2_20140121, 
	't_il2/20131218': til2_20131218, 
	't_misc/20140325': tmisc_20140325, 
	't_misc/20140211': tmisc_20140211,
	'aggre_1U': artf_aggre_1U,
	'aggre_3U': artf_aggre_3U,
	'aggre_10U': artf_aggre_10U
}


def plot_collapse(df_exp, m_df, exps, flag_all=True):
	def flip(items, ncol):
		return itertools.chain(*[items[i::ncol] for i in range(ncol)])
	
	cp = sns.color_palette(n_colors=8)
	for exp in exps:
		col = ['gold', 'lightgreen', 'deepskyblue', 'orchid', 'tan']
		if exp == 't_il2/20140121':
			col = ['gold', 'lightgreen', 'orchid', 'deepskyblue', 'tan']
		conds = np.unique(df_exp[exp]['stim'])
		lst_df = []
		for cond in conds:
			df = m_df[exp][cond].iloc[:,2:]
			new_df = df.copy().dropna(how='all')
			new_df.loc[:,'condition'] = pd.Series([cond]*len(new_df['t_div_0']), index=new_df.index)
			lst_df.append(new_df)
		df = pd.concat(lst_df, ignore_index=True)#.dropna(axis=1, how='all')
		sorted_rows = sorted(range(np.shape(df)[0]), key=lambda x: rank_mean_fam(df, x))
		max_gen = int(max(df_exp[exp][df_exp[exp]['fate']=='died']['gen']))

		t_start = 0.
		fig, ax = plt.subplots()
		for y, i in enumerate(sorted_rows):
			df_fam = df.loc[i]
			t_end = rank_mean_fam(df, i)
			check = int(df_fam['condition'])
			if check == 0 or check == 1:
				ax.plot([t_start, t_end], [y+1.]*2, ls='-', lw=255./len(sorted_rows), c=col[int(df_fam['condition'])], zorder=-1)
			elif check == 2:
				ax.plot([t_start, t_end], [y+1.]*2, ls='-', lw=255./len(sorted_rows), c=col[int(df_fam['condition'])], zorder=-1)
			elif check == 3:
				ax.plot([t_start, t_end], [y+1.]*2, ls='-', lw=255./len(sorted_rows), c=col[int(df_fam['condition'])], zorder=-1)
			elif check == 4:
				ax.plot([t_start, t_end], [y+1.]*2, ls='-', lw=255./len(sorted_rows), c=col[int(df_fam['condition'])], zorder=-1)

			for igen in range(max_gen+1):
				color = np.array([cp[igen]])
				if igen==0:
					label_div = r"$T_{div}^0$"
				else:
					label_div = f"$\Sigma_{{k=0}}^{{g={igen}}} T_{{div}}^k$"
				ax.scatter(df.iloc[i][f't_div_{igen}'], y+1, c=color, marker='o', s=20, lw=0.7, edgecolor='k', label=label_div if y==0 else "")
				ax.scatter(df.iloc[i][f't_death_{igen}'], y+1, c=color, marker='X', s=20, lw=0.5, edgecolor=None, label=f"$T_{{die}}^{igen}$" if y==0 else "")
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(flip(handles, 2), flip(labels, 2), loc='lower right', ncol=2, markerscale=2, frameon=False, handletextpad=0, columnspacing=0, fontsize=14)
		ax.set_xlim(left=0)
		ax.set_ylim(bottom=0, ymax=y+2)

		# experiment specific legend setup
		if exp == 't_il2/20131218':
			g_patch = mpatches.Patch(color=col[1], label='1U IL-2')
			b_patch = mpatches.Patch(color=col[2], label='3U IL-2')
			p_patch = mpatches.Patch(color=col[3], label='10U IL-2')
			leg = fig.legend(handles=[g_patch, b_patch, p_patch], loc='center right', fontsize=12)
		elif exp == 't_il2/20140121':
			g_patch = mpatches.Patch(color=col[1], label='1U IL-2')
			p_patch = mpatches.Patch(color=col[3], label='3U IL-2')
			b_patch = mpatches.Patch(color=col[2], label='10U IL-2')
			leg = fig.legend(handles=[g_patch, p_patch, b_patch], loc='center right', fontsize=12)
		elif exp == 't_misc/20140211':
			p_patch = mpatches.Patch(color=col[3], label='N4')
			b_patch = mpatches.Patch(color=col[2], label=r'N4+$\alpha$CD28')
			t_patch = mpatches.Patch(color=col[4], label='N4+IL-2')
			g_patch = mpatches.Patch(color=col[1], label=r'N4+$\alpha$CD28+IL-2')
			leg = fig.legend(handles=[p_patch, b_patch, t_patch, g_patch], loc='center right', fontsize=12)
		elif exp == 't_misc/20140325':
			t_patch = mpatches.Patch(color=col[4], label='N4')
			p_patch = mpatches.Patch(color=col[3], label=r'N4+$\alpha$CD28')
			g_patch = mpatches.Patch(color=col[1], label='N4+IL-12')
			b_patch = mpatches.Patch(color=col[2], label=r'N4+$\alpha$CD28+IL-12')
			leg = fig.legend(handles=[t_patch, p_patch, g_patch, b_patch], loc='center right', fontsize=12)
		elif exp == 'aggre_IL2':
			g_patch = mpatches.Patch(color=col[1], label='1U')
			p_patch = mpatches.Patch(color=col[2], label='3U')
			b_patch = mpatches.Patch(color=col[3], label='10U')
			leg = fig.legend(handles=[g_patch, p_patch, b_patch], loc='center right', fontsize=12)
			leg.set_title("IL-2 Conc.", prop = {'size': 12})

		plt.xlabel("Time (hour)")
		plt.ylabel("Clone")
		plt.tight_layout(rect=(0, 0, 1, 1))
		if flag_all:
			plt.savefig('out/Fig2-1-clones/cc_all_clones_{0}.pdf'.format(exp.replace('/','-')))
		else:
			plt.savefig('out/Fig2-1-clones/cc_filtered_{0}.pdf'.format(exp.replace('/','-')))
		plt.close()


def plot_cascade(df_exp, exps):
	def rank_order(data, sortby):
		df = data[data['t_death'].notna()].loc[:,['clone', 'gen', sortby]].copy()
		df = df.sort_values(by=sortby, ascending=False)

		ranked_clone = df["clone"].to_list()
		indices = np.unique(ranked_clone, return_index=True)[1]
		ranked_clone = [ranked_clone[index] for index in sorted(indices)]

		return df, ranked_clone
		
	cp = sns.color_palette(n_colors=8)  # fixed colors for generation
	for exp in exps:
		exp_lab = exp.replace('/', '_')
		conditions = np.unique(df_exp[exp]['stim'])
		for cond in conditions:
			data = df_exp[exp][df_exp[exp]['stim'] == cond]

			max_gen = int(max(data['gen']))

			#### FILTER DATA
			unique_clone = np.unique(data['clone'])

			complete, frayed = [], []
			filtered = []  # final decision: complete + frayed families whose death is the last event (i.e. remove End of Movie families)
			for uc in unique_clone:
				clone = data[data['clone'] == uc]
				max_gen = int(max(clone['gen']))
				
				## Determine complete tree by number of cells observed in each clone
				true_false_table = []
				for igen in range(max_gen+1):
					hypo_num_cells = 2**igen
					curr_num_cells = len(clone[clone['gen'] == igen])
					true_false_table.append(hypo_num_cells == curr_num_cells)
				true_false_table = np.array(true_false_table)
				if np.all(true_false_table): complete.append(uc)
				else: frayed.append(uc)
				
				## Get times
				lost_times = np.array(clone['t_loss'])
				death_times = np.array(clone['t_death'])
				div_times = np.array(clone['t_div'])
				number_cells = len(clone)
				number_lost_cells = len(lost_times[~np.isnan(lost_times)])
				max_lost_time = np.nanmax(lost_times)
				max_death_time = np.nanmax(death_times)
				max_div_time = np.nanmax(div_times)

				## FILTER RULE 
				# 1. At lesat divided once
				# 2. Clones whose last event is death, not division or lost
				if (number_cells > 1) and \
					(number_lost_cells > 0 and max_death_time > max_lost_time) or \
					(number_lost_cells == 0 and max_death_time > max_div_time):
					filtered.append(uc)

			# complete_df = data[data['clone'].isin(complete)].copy()
			# frayed_df = data[data['clone'].isin(frayed)].copy()
			filtered_df = data[data['clone'].isin(filtered)].copy()

			total_clones = len(unique_clone)
			num_complete = len(complete); perc_complete = num_complete/total_clones*100
			num_frayed = len(frayed); perc_frayed = num_frayed/total_clones*100
			num_final = len(filtered); perc_final = num_final/total_clones*100
			print(f"\n[{exp}] [{cond_labs[exp][cond]}] consists of...")
			print(f" >> Total {total_clones} clones...")
			print(f" > Complete family: {num_complete} clones ({perc_complete:.2f}%)")
			print(f" > Frayed family: {num_frayed} clones ({perc_frayed:.2f}%)")
			print(f" > [Selected] Complete + Frayed without End of Movie family: {num_final} clones ({perc_final:.2f}%)\n")

			#### Cascade Plot
			c_df_tld, _ = rank_order(filtered_df, 't_birth')
			c_df_die, _ = rank_order(filtered_df, 't_death')
			if num_final and len(c_df_tld) and len(c_df_die):
				max_gen = int(max(filtered_df['gen']))
				figS1, axS1 = plt.subplots(nrows=2, ncols=2, figsize=(16, 5), sharey='row')  # Style 1 (S1)
				axS1 = axS1.reshape(2,2)
				figS2, axS2 = plt.subplots(nrows=4, ncols=1, figsize=(8, 10), sharey='row')  # Style 2 (S2)
				axS2 = axS2.reshape(2,2)

				tdiv0_list = []  # store the time to first division
				clone_list = []
				gen_list = []
				tdiv_list = []  # store the average subsequent division time per clone per generation
				avg_tdiv_list = []; cv_tdiv_list = []
				avg_tdie_list = []; cv_tdie_list = []
				avg_tld_list = []; cv_tld_list = []
				uniq_clones = np.unique(filtered_df['clone'])
				for cl in uniq_clones:
					tdiv0 = filtered_df[(filtered_df.clone==cl) & (filtered_df.fate=='divided') & (filtered_df.gen==0)].t_div.to_numpy() # time to first division
					if len(tdiv0) > 0: tdiv0_list.append(tdiv0[0])
					else: tdiv0_list.append(np.nan)  # if no data put NaN

					avg_tdiv = filtered_df[(filtered_df.clone==cl) & (filtered_df.fate=='divided') & (filtered_df.gen>0)].lifetime.to_numpy()  # average subsequent division time
					if len(avg_tdiv) > 0: avg_tdiv_list.append(np.nanmean(avg_tdiv))
					else: avg_tdiv_list.append(np.nan)

					tmp_tdiv_list = []
					mgenn = int(max(filtered_df[filtered_df.clone==cl].gen))
					for igen in range(1, mgenn+1):
						tdiv_cells = filtered_df[(filtered_df.clone==cl) & (filtered_df.fate=='divided') & (filtered_df.gen==igen)].lifetime.to_numpy()
						# tdiv = np.mean(tdiv_cells)
						if len(tdiv_cells) > 0:
							for val_tdiv in tdiv_cells:
								clone_list.append(cl)
								gen_list.append(igen)
								tdiv_list.append(val_tdiv)
								tmp_tdiv_list.append(val_tdiv)
					if len(tmp_tdiv_list) == 1:
						cv_tdiv_list.append(0)
					elif len(tmp_tdiv_list) > 1:
						cv_tdiv_list.append(np.nanstd(tmp_tdiv_list, ddof=1)/np.mean(tmp_tdiv_list))
					else:
						cv_tdiv_list.append(np.nan)

					dead = filtered_df[(filtered_df.clone==cl) & (filtered_df.fate=='died')].t_death.to_numpy()
					if len(dead) == 1:
						avg_tdie = np.nanmean(dead)
						avg_tdie_list.append(avg_tdie)
						cv_tdie_list.append(0)
					elif len(dead) > 1:
						avg_tdie = np.nanmean(dead)
						avg_tdie_list.append(avg_tdie)
						cv_tdie_list.append(np.nanstd(dead, ddof=1)/avg_tdie)
					else:
						avg_tdie_list.append(np.nan)
						cv_tdie_list.append(np.nan)

					cell_rel = np.array(filtered_df[(filtered_df.clone==cl) & (filtered_df.fate=='died')].relation)
					cell_rel_prec = [label[:-1] for label in cell_rel]  # determine the relation in the family tree
					y_clo = np.array(filtered_df[(filtered_df.clone==cl) & (filtered_df.fate=='divided') & (filtered_df.relation.isin(cell_rel_prec))].t_div)
					y_clo = y_clo[~np.isnan(y_clo)]
					if len(y_clo) == 1:
						avg_tld = np.nanmean(y_clo)
						avg_tld_list.append(avg_tld)
						cv_tld_list.append(0)
					elif len(y_clo) > 1:
						avg_tld = np.nanmean(y_clo)
						avg_tld_list.append(avg_tld)
						cv_tld_list.append(np.nanstd(y_clo, ddof=1)/avg_tld)
					else:
						avg_tld_list.append(np.nan)
						cv_tld_list.append(np.nan)

				######## First division
				df_tdiv0 = pd.DataFrame({
					"clone": uniq_clones,
					"tdiv0": tdiv0_list
				})
				df_tdiv0 = df_tdiv0[['clone', 'tdiv0']].dropna()
				df_tdiv0 = df_tdiv0.sort_values(by="tdiv0", ascending=False)
				ranked_tdiv0 = df_tdiv0['clone'].to_list()
				indices = np.unique(ranked_tdiv0, return_index=True)[1]
				ranked_tdiv0 = [ranked_tdiv0[index] for index in sorted(indices)]

				######## Subsequent division
				df_tdiv = pd.DataFrame({
					"clone": clone_list,
					"gen": gen_list,
					"tdiv": tdiv_list
				})
				df_tdiv = df_tdiv[['clone', 'gen', 'tdiv']].dropna()
				df_tdiv = df_tdiv.sort_values(by="tdiv", ascending=False)
				ranked_tdiv = df_tdiv['clone'].to_list()
				indices = np.unique(ranked_tdiv, return_index=True)[1]
				ranked_tdiv = [ranked_tdiv[index] for index in sorted(indices)]
				if len(df_tdiv["gen"]) > 0:
					tdiv_min_gen = int(min(df_tdiv["gen"]))
					df_tdiv['gen'] = df_tdiv['gen'].astype(int)

					df_avg_tdiv = pd.DataFrame({
						"clone": uniq_clones,
						"avg_tdiv": avg_tdiv_list,
						'cv_tdiv': cv_tdiv_list
					})
					df_avg_tdiv = df_avg_tdiv[['clone', 'avg_tdiv', 'cv_tdiv']].dropna()
					df_avg_tdiv = df_avg_tdiv.sort_values(by="avg_tdiv", ascending=False)
					ranked_avg_tdiv = df_avg_tdiv['clone'].to_list()
					indices = np.unique(ranked_avg_tdiv, return_index=True)[1]
					ranked_avg_tdiv = [ranked_avg_tdiv[index] for index in sorted(indices)]

				######## Last division
				df_avg_tld = pd.DataFrame({
					"clone": uniq_clones,
					"avg_tld": avg_tld_list,
					'cv_tld': cv_tld_list
				})
				df_avg_tld = df_avg_tld[['clone', 'avg_tld', 'cv_tld']].dropna()
				df_avg_tld = df_avg_tld.sort_values(by="avg_tld", ascending=False)
				ranked_avg_tld = df_avg_tld['clone'].to_list()
				indices = np.unique(ranked_avg_tld, return_index=True)[1]
				ranked_avg_tld = [ranked_avg_tld[index] for index in sorted(indices)]

				tld_min_gen = int(min(c_df_tld["gen"]))
				c_df_tld['gen'] = c_df_tld['gen'].astype(int)

				######## Death
				df_avg_tdie = pd.DataFrame({
					"clone": uniq_clones,
					"avg_tdie": avg_tdie_list,
					'cv_tdie': cv_tdie_list
				})
				df_avg_tdie = df_avg_tdie[['clone', 'avg_tdie', 'cv_tdie']].dropna()
				df_avg_tdie = df_avg_tdie.sort_values(by="avg_tdie", ascending=False)
				ranked_avg_tdie = df_avg_tdie['clone'].to_list()
				indices = np.unique(ranked_avg_tdie, return_index=True)[1]
				ranked_avg_tdie = [ranked_avg_tdie[index] for index in sorted(indices)]

				die_min_gen = int(min(c_df_die["gen"]))
				c_df_die['gen'] = c_df_die['gen'].astype(int)

				for iax, ax in enumerate([axS1, axS2]):
					sns.stripplot(x='clone', 
								  y='tdiv0', 
								  data=df_tdiv0, 
								  order=ranked_tdiv0, 
								  palette=cp[:1], 
								  label="0", 
								  ax=ax[0][0])
					if len(df_tdiv["gen"]) > 0:
						sns.stripplot(x='clone', 
									  y='tdiv', 
									  hue='gen', 
									  data=df_tdiv, 
									  order=ranked_avg_tdiv, 
									  palette=cp[tdiv_min_gen:], 
									  zorder=1, 
									  ax=ax[0][1])
						ax[0][1].annotate(f"Avg. CV: {df_avg_tdiv['cv_tdiv'].mean()*100:.1f}%", xy=(0.02, 1.0), xycoords="axes fraction", weight='bold')
					else: 
						ax[0][1].annotate("NA", xy=(0.45, 0.5), xycoords='axes fraction', weight='bold', fontsize=24)
						ax[0][1].set_xticklabels([])
						ax[0][1].set_yticklabels([])
					sns.stripplot(x='clone', 
								  y='t_birth', 
								  hue='gen', 
								  data=c_df_tld, 
								  order=ranked_avg_tld, 
								  palette=cp[tld_min_gen:], 
								  jitter=True, 
								  zorder=1, 
								  ax=ax[1][0])
					ax[1][0].annotate(f"Avg. CV: {df_avg_tld['cv_tld'].mean()*100:.1f}%", xy=(0.02, 1.0), xycoords="axes fraction", weight='bold')
					sns.stripplot(x='clone', 
								  y='t_death', 
								  hue='gen', 
								  data=c_df_die, 
								  marker='X', 
								  order=ranked_avg_tdie, 
								  palette=cp[die_min_gen:], 
								  jitter=True, 
								  zorder=1, 
								  ax=ax[1][1])
					ax[1][1].annotate(f"Avg. CV: {df_avg_tdie['cv_tdie'].mean()*100:.1f}%", xy=(0.02, 1.0), xycoords="axes fraction", weight='bold')

					if len(df_tdiv0['tdiv0']) > 0:
						ax[0][0].set_title(r"Time to first division ($T_{div}^0$)")
						ax[0][0].set_ylabel('Time (hour)')
						ax[0][0].set_xlabel("")
						ax[0][0].grid(True, which='major', axis='both', linestyle='--')
						ax[0][0].set_xticklabels([]) # ax[0][0].set_xticklabels(ax[0][0].get_xticklabels(), fontsize=10.5, rotation=90)
						ax[0][0].set_ylim(bottom=0)

						ax[0][1].set_title(r"Subsequent division time ($T_{div}^k$)")
						if len(df_tdiv["gen"]) > 0:
							if iax == 0: ax[0][1].set_ylabel("")
							elif iax == 1: ax[0][1].set_ylabel('Time (hour)')
						else:
							ax[0][1].set_ylabel("")
						ax[0][1].set_xlabel("")
						ax[0][1].grid(True, which='major', axis='both', linestyle='--')
						ax[0][1].set_xticklabels([]) # ax[0][1].set_xticklabels(ax[0][1].get_xticklabels(), fontsize=10.5, rotation=90)
						ax[0][1].set_ylim(bottom=0)
						if len(df_tdiv['tdiv']) > 0:
							ax[0][1].legend_.remove()

					ax[1][0].set_title(r"Time to last division ($T_{ld}$)")
					ax[1][0].set_ylabel('Time (hour)')
					if iax == 0: ax[1][0].set_xlabel('Clone')
					elif iax == 1: ax[1][0].set_xlabel("")
					ax[1][0].grid(True, which='major', axis='both', linestyle='--')
					ax[1][0].set_xticklabels([]) # ax[1][0].set_xticklabels(ax[1][0].get_xticklabels(), fontsize=10.5, rotation=90)
					ax[1][0].set_ylim(bottom=0)
					ax[1][0].legend_.remove()

					ax[1][1].set_title(r"Time to death ($T_{die}$)")
					if iax == 0: ax[1][1].set_ylabel("")
					elif iax == 1: ax[1][1].set_ylabel('Time (hour)')
					ax[1][1].set_xlabel('Clone')
					ax[1][1].grid(True, which='major', axis='both', linestyle='--')
					ax[1][1].set_xticklabels([]) # ax[1][1].set_xticklabels(ax[1][1].get_xticklabels(), fontsize=10.5, rotation=90)
					ax[1][1].set_ylim(bottom=0)
					h0, l0 = ax[0][0].get_legend_handles_labels()
					h1, l1 = ax[1][1].get_legend_handles_labels()
					leg = ax[0][0].legend(loc='lower left', handles=list(np.append(h0[:1], h1)), labels=list(np.append(l0[:1], l1)), ncol=max_gen+1, frameon=True, columnspacing=1, handletextpad=0.1, fontsize=16)
					for lh in leg.legendHandles:
						lh._sizes = [40]
					leg.set_title("Generation", prop = {'size': 16})
					ax[1][1].legend_.remove()
				figS1.tight_layout(rect=(0, 0, 1, 1))
				figS1.subplots_adjust(hspace=0.2, wspace=0.02)
				figS1.savefig(f"./out/Fig2-2-raw/Style1/{exp_lab}-{cond_labs[exp][cond]}_S1_cascade.pdf", dpi=300)

				figS2.tight_layout(rect=(0, 0, 1, 1))
				figS2.subplots_adjust(hspace=0.2, wspace=0.02)
				figS2.savefig(f"./out/Fig2-2-raw/Style2/{exp_lab}-{cond_labs[exp][cond]}_S2_cascade.pdf", dpi=300)


def corr(df_exp, exps):
	def dist(x, **kws):
		ax = plt.gca()
		kws['color'] = next(colors)
		ax.set_title(f"{next(titles)}", color=kws['color'], fontsize=16)
		if len(x) > 0:
			bin_list = np.arange(start=0, stop=np.nanmax(x)+1, step=1)
			sns.distplot(x, bins=bin_list, kde=False, norm_hist=True, color=kws['color'], ax=ax)
			ax.annotate(f"$N$ = {len(x)}", xy=(0.02, 0.88), xycoords=ax.transAxes, color=kws['color'], fontsize=16)
		else:
			sns.distplot(x, kde=False, norm_hist=True, color=kws['color'], ax=ax)
			ax.annotate("NA", xy=(0.02, 0.88), xycoords=ax.transAxes, color=kws['color'], fontsize=16)
		ax.set_xlim(left=0)

	def scatter(x, y, **kws):
		def _covariance(sigma, rho):
			C = T.fill_diagonal(T.alloc(rho, 2, 2), 1.)
			S = T.diag(sigma)
			return S.dot(C).dot(S)

		print(f'------------------------ BEGIN ({x.name}, {y.name}) ------------------------')
		ax = plt.gca()
		ax.spines['top'].set_visible(True)
		ax.spines['right'].set_visible(True)

		### SET ITERATION AND TUNING NUMBERS
		niter, ntune = 100000, 10000
		nchain = ncore = 5
		if len(x) > 2 and len(y) > 2:
			data = np.array([x, y]).T
			data = data[~np.isnan(data).any(axis=1)]  # remove nan
			n, _ = np.shape(data)
			with pm.Model() as model:
				## Multivariate normal
				mu = pm.Uniform('mu', lower=1E-6, upper=1E3, shape=2)
				sigma = pm.Uniform('sigma', lower=1E-6, upper=1E3, shape=2)
				rho = pm.Uniform('rho', lower=-1, upper=1)
				cov = pm.Deterministic('cov', _covariance(sigma, rho))
				mvn = pm.MvNormal('mvn', mu=mu, cov=cov, observed=data)
				trace = pm.sample(draws=niter, 
								  tune=ntune, 
								  chains=nchain, 
								  cores=ncore,
								  random_seed=57323343,
								  target_accept=0.95, 
								  init='jitter+adapt_diag')
			summary = pm.summary(trace)

			mu_post = trace['mu'].mean(axis=0)
			cov_post = trace['cov'].mean(axis=0)
			var_post, u_post = np.linalg.eig(cov_post)
			angle_post = np.degrees(np.arctan2(*u_post[:,0][::-1]))

			## 95% Credible range
			s = 5.991
			width, height = 2*np.sqrt(s*var_post[0]), 2*np.sqrt(s*var_post[1])
			e_post1 = mpatches.Ellipse(xy=mu_post, width=width, height=height, angle=angle_post, facecolor='black', alpha=0.2, zorder=0)
			ax.add_artist(e_post1)

			## 90% Credible range
			s = 4.605
			width, height = 2*np.sqrt(s*var_post[0]), 2*np.sqrt(s*var_post[1])
			e_post2 = mpatches.Ellipse(xy=mu_post, width=width, height=height, angle=angle_post, facecolor='none', edgecolor='black', linestyle='--', zorder=0)
			ax.add_artist(e_post2)

			## 99% Credible range
			s = 9.210
			width, height = 2*np.sqrt(s*var_post[0]), 2*np.sqrt(s*var_post[1])
			e_post3 = mpatches.Ellipse(xy=mu_post, width=width, height=height, angle=angle_post, facecolor='none', edgecolor='black', linestyle=':', zorder=0)
			ax.add_artist(e_post3)

			rho_post = trace['rho'].mean(axis=0)
			hpd = pm.stats.hpd(trace['rho'], alpha=0.05)
			ax.annotate(r"$\rho$" + f" = {rho_post:.2f} [{hpd[0]:.2f}, {hpd[1]:.2f}]", xy=(0.02, 0.05), xycoords=ax.transAxes, fontsize=16)

			## LOCATION OF MEANS
			# ax.axvline(mu_post[0], color='k', linestyle='-')
			# ax.axhline(mu_post[1], color='k', linestyle='-')

			### CALCULATE BAYESIAN FACTOR
			## Estimate from posterior distribution of rho by using Savage-Dickey density ratio method (CH.13 PP.179 Bayesian Cognitive Modeling)
			## https://github.com/pymc-devs/resources/blob/master/BCM/CaseStudies/ExtrasensoryPerception.ipynb
			posterior = sps.gaussian_kde(trace['rho'])(0)[0]
			prior = sps.uniform.pdf(0, loc=-1., scale=2.)  # Ranges [loc, loc + scale]; Technically it's always 0.5
			BayesFactor01 = posterior/prior

			## Approximate Jeffreys (1961), pp. 289-292:
			sample_r = np.corrcoef(data[:,0], data[:,1])  # sample correlation
			BayesFactor_approx = 1 / (((2 * (n - 1) - 1) / np.pi) ** 0.5 * (1 - sample_r[0, 1] ** 2) ** (0.5 * ((n - 1) - 3)))  # BF10

			## Exact solution Jeffreys (numerical integration) Theory of Probability (1961), pp. 291 Eq.(9):
			## Or nicely presented in Wagenmakers et al. 2016 Appendix
			f_int = lambda rho: ((1. - rho ** 2.) ** ((n - 1.) / 2.)) / ((1. - rho * sample_r[0, 1]) ** (n - 3./2.))
			BayesFactor_exact = 0.5 * sp.integrate.quad(f_int, -1, 1)[0]
			print("\n>> BF10 (BF01):")
			print(f"    Estimate from posterior: {1/BayesFactor01:.5f} ({BayesFactor01:.5f})")
			print(f"    Approximate Jeffreys   : {BayesFactor_approx:.5f} ({1/BayesFactor_approx:.5f})")
			print(f"    Exact solution         : {BayesFactor_exact:.5f} ({1/BayesFactor_exact:.5f})\n")
			if BayesFactor_exact > 1:  # BF10: Favours the alternative hypothesis (rho != 0)
				if BayesFactor_exact == 1.:
					string = r"BF$_{10}$" + f" = {BayesFactor_exact:.2f}"; color = "#000000"
				elif 1 < BayesFactor_exact < 3:
					string = r"BF$_{10}$" + f" = {BayesFactor_exact:.2f}"; color = "#000000"
				elif 3 < BayesFactor_exact < 10:
					string = r"BF$_{10}$" + f" = {BayesFactor_exact:.2f}"; color = "#400000"
				elif 10 < BayesFactor_exact < 30:
					string = r"BF$_{10}$" + f" = {BayesFactor_exact:.2f}"; color = "#800000"
				elif 30 < BayesFactor_exact < 100:
					string = r"BF$_{10}$" + f" = {BayesFactor_exact:.2f}"; color = "#BF0000"
				elif BayesFactor_exact > 100:
					string = r"BF$_{10}$" + f" > 100"; color = "#FF0000"
			else:  # BF01: Favours the null hypothesis (rho = 0)
				BayesFactor01_exact = 1/BayesFactor_exact
				if BayesFactor01_exact == 1.:
					string = r"BF$_{01}$" + f" = {BayesFactor01_exact:.2f}"; color = "#000000"
				elif 1 < BayesFactor01_exact < 3:
					string = r"BF$_{01}$" + f" = {BayesFactor01_exact:.2f}"; color = "#000000"
				elif 3 < BayesFactor01_exact < 10:
					string = r"BF$_{01}$" + f" = {BayesFactor01_exact:.2f}"; color = "#000080"
				elif 10 < BayesFactor01_exact < 30:
					string = r"BF$_{01}$" + f" = {BayesFactor01_exact:.2f}"; color = "#061D95"
				elif 30 < BayesFactor01_exact < 100:
					string = r"BF$_{01}$" + f" = {BayesFactor01_exact:.2f}"; color = "#0D3AA9"
				elif BayesFactor01_exact > 100:
					string = r"BF$_{01}$" + f" > 100"; color = "#1974D2"
			ax.annotate(string, xy=(0.02, 0.18), xycoords=ax.transAxes, color=color, fontsize=16)
			print(summary, end='\n\n')
		else:
			ax.annotate("NA", xy=(0.02, 0.05), xycoords=ax.transAxes, fontsize=16)
		ax.set_xlim(left=0)
		ax.set_ylim(bottom=0)

	excl = ['t_il2/20131218', 't_il2/20140121']
	exps = [exp for exp in exps if exp not in excl]  # exclude individual IL-2 dataset (aggregated one separtely done)

	print("\n>> CORRELATION ANALYSIS...")
	for exp in exps:
		print(f'\n======================== BEGIN {exp} ========================')
		exp_lab = exp.replace('/', '_')
		conds = np.unique(list(df_exp[exp].keys()))  # find unique conditions
		for cond in conds:
			print(f'======================== BEGIN {cond_labs[exp][cond]} ========================')
			titles = iter([r'Time to first div ($T_{div}^0$)', r'Avg. sub div time ($T_{div}^{k\geq1}$)', 
							r'Time to last div ($T_{ld}$)', r'Time to death ($T_{die}$)'])
			colors = iter(['blue', 'orange', 'green', 'red'])

			df = df_exp[exp][cond].copy() 

			CONDS = []; CONDS.append(cond)
			TIME_TO_FIRST_DIV = []
			AVG_SUB_DIV = []
			TIME_TO_LAST_DIV = []
			TIME_TO_DEATH = []

			clones = np.unique(df['clone'])
			for cl in clones:
				tdie = np.nanmean(df[(df.clone==cl) & (df.fate=='died')].t_death)
				tdiv0 = df[(df.clone==cl) & (df.fate=='divided') & (df.gen==0)].t_div.to_numpy() # time to first division
				tdiv = np.nanmean(df[(df.clone==cl) & (df.fate=='divided') & (df.gen>0)].lifetime)  # average subsequent division time

				cell_rel = np.array(df[(df.clone==cl) & (df.fate=='died')].relation)
				cell_rel_prec = [label[:-1] for label in cell_rel]  # determine the relation in the family tree
				y_clo = np.array(df[(df.clone==cl) & (df.fate=='divided') & (df.relation.isin(cell_rel_prec))].t_div)
				y_clo = y_clo[~np.isnan(y_clo)]
				tld = np.nanmean(y_clo)

				if len(tdiv0) > 0:
					TIME_TO_FIRST_DIV.append(tdiv0[0])
				else:
					TIME_TO_FIRST_DIV.append(np.nan)  # if no data put NaN
				AVG_SUB_DIV.append(tdiv)
				TIME_TO_DEATH.append(tdie)
				TIME_TO_LAST_DIV.append(tld)

			df_pair = pd.DataFrame({
				"$T_{div}^0$": TIME_TO_FIRST_DIV,
				"$T_{div}^{k\geq1}$": AVG_SUB_DIV,
				"$T_{ld}$": TIME_TO_LAST_DIV,
				"$T_{die}$": TIME_TO_DEATH
			})
			plt.rcParams.update({'axes.titlesize': 22, 'axes.labelsize': 22})
			g = sns.pairplot(df_pair, height=1.5, corner=True, diag_kind='None', 
							 plot_kws={'s': 22, 'fc': "grey", 'ec': 'k', 'linewidth': 1})
			g.map_diag(dist)
			g.map_lower(scatter)
			g.fig.set_size_inches(12, 9)
			plt.rcParams.update({'axes.titlesize': rc['axes.titlesize'], 
								'ytick.labelsize': rc['ytick.labelsize']})

			## Add Bayes Factor interpretation scale
			bf_bounds = [1, 3, 10, 30, 100]
			ax2 = g.fig.add_axes(rect=[0.56, 0.90, 0.41, 0.05])
			bf10_colors = ["#000000", "#000080", "#061D95", "#0D3AA9"]
			cmap = mpl.colors.ListedColormap(bf10_colors)
			cmap.set_over("#1974D2")
			norm = mpl.colors.BoundaryNorm(bf_bounds, cmap.N)
			cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, boundaries=bf_bounds+[105], extend='max', ticks=bf_bounds, spacing='uniform', orientation='horizontal')
			cb.ax.set_title(r"BF$_{01}$: In favour of $H_0$ ($\rho = 0$)", weight='bold')
			ax2.set_xticklabels(["1", "3", "10", "30", ">100"])

			ax1 = g.fig.add_axes(rect=[0.56, 0.78, 0.41, 0.05])
			bf01_colors = ["#000000", "#400000", "#800000", "#BF0000"]
			cmap = mpl.colors.ListedColormap(bf01_colors)
			cmap.set_over("#FF0000")
			norm = mpl.colors.BoundaryNorm(bf_bounds, cmap.N)
			cb = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, boundaries=bf_bounds+[105], extend='max', ticks=bf_bounds, spacing='uniform', orientation='horizontal')
			cb.ax.set_title(r"BF$_{10}$: In favour of $H_1$ ($\rho\neq 0$)", weight='bold')
			cb.set_label(r"Anecdotal (1<BF<3); Moderate (3<BF<10)" + "\nStrong (10<BF<30); Very strong (30<BF<100)" + "\nExtreme (BF>100)", fontsize=16)
			ax1.set_xticklabels(["1", "3", "10", "30", ">100"])

			g.fig.tight_layout(rect=(0, 0, 1, 1))
			g.fig.subplots_adjust(hspace=0.07, wspace=0.07)
			g.fig.savefig(f"./out/Fig2-3-corr/{exp_lab}-{cond_labs[exp][cond]}_corr.pdf", dpi=300)

if __name__ == "__main__":
	data_path = './data'
	exp_paths = [os.path.join(data_path, f'{exp}') for exp in os.listdir(data_path) if not exp.startswith('.') and exp != '_processed']
	exp_labs = [os.path.join(exp, lab) for exp in exp_paths for lab in os.listdir(exp) if not lab.startswith('.')]

	#### Organise data
	exps, df_exp, df_CC = parse_data(exp_labs)  # returns keys, raw data and clonally collapsed data frames
	for exp in exps:
		conds = np.unique(df_exp[exp]['stim'])
		for c in conds:
			print(f"[{exp}]-[{cond_labs[exp][c]}] {len(np.unique(df_exp[exp][df_exp[exp]['stim']==c]['clone']))} clones")

	#### Filter data
	df_F, df_F_CC = filter_data(df_exp, exps)  # clonal collapse data (mean of time to divide and die in each generation)

	#### SAVE DATAFRAMES
	save_dataframes(exps, df_exp, df_CC, df_F)
	save_cc_times(exps, df_F)

	#### Plot filming data
	plot_collapse(df_exp, df_CC, exps, flag_all=True)
	plot_collapse(df_exp, df_F_CC, exps, flag_all=False)
	plot_cascade(df_exp, exps)

	#### Correlation analysis
	corr(df_F, exps)


	#### AGGREGATE T CELL IL-2 DATA (WITH SAME CONCENTRATION: 1U, 3U, 10U)
	print("\nAggregating IL-2 experiments...")
	def rename_clone(prev_max, df, ff=True):
		unique_clone = np.unique(df['clone'])
		new_unique_clone, counter = [], 1
		for _ in unique_clone:
			new_unique_clone.append(prev_max + counter)
			counter += 1

		## MAP UNIQUE CLONE TO ANOTHER UNIQUE (INCREMENTAL) LABEL
		tmp_df = df.copy()
		for uc, new_uc in zip(unique_clone, new_unique_clone):
			idx = tmp_df.index[tmp_df['clone']==uc].to_numpy()
			df.loc[idx, 'clone'] = new_uc
		return df

	il2_20131218 = df_exp[exps[exps.index('t_il2/20131218')]]
	il2_20140121 = df_exp[exps[exps.index('t_il2/20140121')]]

	## Select 1U
	a1U = il2_20131218[il2_20131218['stim']==1.0].copy()
	print("20131218-1U: #clones", len(np.unique(a1U['clone'])))
	b1U = il2_20140121[il2_20140121['stim']==1.0].copy()
	print("20140121-1U: #clones", len(np.unique(b1U['clone'])))
	b1U = rename_clone(max(a1U['clone']), b1U)  # update the clone label
	aggre_1U = pd.concat([a1U, b1U]).reset_index(drop=True)
	aggre_1U['stim'] = 1.0
	concat1 = aggre_1U.copy()

	## Select 3U
	a3U = il2_20131218[il2_20131218['stim']==2.0].copy()
	print("20131218-3U: #clones", len(np.unique(a3U['clone'])))
	b3U = il2_20140121[il2_20140121['stim']==3.0].copy()
	print("20140121-3U: #clones", len(np.unique(b3U['clone'])))
	# relabel clone id for the first df
	a3U = rename_clone(-1, a3U)
	b3U = rename_clone(max(a3U['clone']), b3U)  # update the clone label	
	aggre_3U = pd.concat([a3U, b3U]).reset_index(drop=True)
	aggre_3U['stim'] = 2.0
	concat2 = aggre_3U.copy()
	concat2 = rename_clone(max(concat1['clone']), concat2)

	## Select 10U
	a10U = il2_20131218[il2_20131218['stim']==3.0].copy()
	print("20131218-10U: #clones", len(np.unique(a10U['clone'])))
	b10U = il2_20140121[il2_20140121['stim']==2.0].copy()
	print("20140121-10U: #clones", len(np.unique(b10U['clone'])))
	# relabel clone id for the first df
	a10U = rename_clone(-1, a10U)
	b10U = rename_clone(max(a10U['clone']), b10U)
	aggre_10U = pd.concat([a10U, b10U]).reset_index(drop=True)
	aggre_10U['stim'] = 3.0
	concat3 = aggre_10U.copy()
	concat3 = rename_clone(max(concat2['clone']), concat3)

	concat = pd.concat([concat1, concat2, concat3]).copy()
	df_concat = {'aggre_IL2': concat}

	print(f"Aggre 1U: #clones {len(np.unique(concat[concat['stim']==1.0]['clone']))}")
	print(f"Aggre 3U: #clones {len(np.unique(concat[concat['stim']==2.0]['clone']))}")
	print(f"Aggre 10U: #clones {len(np.unique(concat[concat['stim']==3.0]['clone']))}")
	print(f"Aggre Total: #clone {len(np.unique(concat['clone']))}")

	# Clonal collapse (ignore lost cells)
	keys = ['aggre_IL2']
	df_aggre_CC = {exp: {cond: {} for cond in np.unique(concat['stim'])} for exp in keys}
	for exp in keys:
		conds = np.unique(concat['stim'])
		for cond in conds:
			df = concat[concat['stim']==cond]  # select all data == cond
			unique_clone = np.unique(df['clone'])
			n_fams = len(unique_clone)
			
			max_gen = int(max(df['gen']))
			collapse = {'clone': [], 'stim': [cond]*n_fams}
			for igen in range(max_gen+1):
				collapse[f't_div_{igen}'] = []
				collapse[f't_death_{igen}'] = []

			for uc in unique_clone:
				collapse['clone'].append(uc)
				clone = df[df['clone']==uc].copy()
				for igen in range(max_gen+1):
					df_gen = clone[clone['gen']==igen]
					if len(df_gen) != 0:  # check if data is empty at igen
						# collect division times
						t_div = np.array(df_gen[~np.isnan(df_gen['t_div'])]['t_div'])
						if len(t_div) != 0:
							mean_t_div = np.mean(t_div)
							collapse[f't_div_{igen}'].append(mean_t_div)
						else:  # no division time information
							collapse[f't_div_{igen}'].append(np.nan)

						# collect death times
						t_death = np.array(df_gen[~np.isnan(df_gen['t_death'])]['t_death'])
						if len(t_death) != 0:
							mean_t_death = np.mean(t_death)
							collapse[f't_death_{igen}'].append(mean_t_death)
						else:  # no death time information
							collapse[f't_death_{igen}'].append(np.nan)
					else:
						collapse[f't_div_{igen}'].append(np.nan)
						collapse[f't_death_{igen}'].append(np.nan)
			df_aggre_CC[exp][cond] = pd.DataFrame(collapse)
			df_aggre_CC[exp][cond] = df_aggre_CC[exp][cond].dropna(how='all')

	df_exp_concat = {'aggre_IL2': pd.concat([aggre_1U, aggre_3U, aggre_10U])}
	df_F, df_F_CC = filter_data(df_exp_concat, keys)
	plot_collapse(df_exp_concat, df_aggre_CC, keys, flag_all=True)
	plot_collapse(df_exp_concat, df_F_CC, keys, flag_all=False)

	exps = ['aggre_1U', 'aggre_3U', 'aggre_10U']
	df_il2 = {'aggre_1U': aggre_1U, 'aggre_3U': aggre_3U, 'aggre_10U': aggre_10U}
	df_F, df_F_CC = filter_data(df_il2, exps)
	save_dataframes(exps, df_il2, df_F_CC, df_F)
	save_cc_times(exps, df_F)

	plot_cascade(df_il2, exps)

	corr(df_F, exps)
