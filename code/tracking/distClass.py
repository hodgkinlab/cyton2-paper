"""
Last update: 30-October-2020

Bayesian inference to select the best distribution class (i.e. model selection) given the observed times
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sps
import pymc3 as pm
import xarray as xr
rng = np.random.RandomState(seed=61114724)
# pd.set_option('display.max_rows', 999999)

from _func import ecdf

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
	'lines.markersize': 7.5, 'lines.linewidth': 1,
	'errorbar.capsize': 2.5
}
sns.set(context='paper', style='white', rc=rc)

def plot_traces(traces, retain=0):
	ax = pm.traceplot(traces[-retain:], lines=tuple([(k, {}, v['mean']) for k, v in pm.summary(traces[-retain:]).iterrows()]))

	for i, mn in enumerate(pm.summary(traces[-retain:])['mean']):
		ax[i, 0].annotate('{:.2f}'.format(mn), xy=(mn, 0), xycoords='data', xytext=(5, 10), textcoords='offset points', rotation=90, va='bottom', fontsize='large', color='#AA0022')

def confidence_band(x):
	mean = np.mean(x, axis=0)
	low = np.percentile(x, 2.5, interpolation='nearest', axis=0)
	high = np.percentile(x, 97.5, interpolation='nearest', axis=0)
	return mean, low, high

VAR_MAP = {'tdiv0': "$T_{div}^0$", 'tdiv': "$M$", 'tld': "$T_{ld}$", 'tdie': "$T_{die}$"}
DATA_PROPERTY = {
	'./data/_processed/collapsed_times/b_cpg_cpg3_0.0.csv': {
		'condition': "CpG"
	},
	'./data/_processed/collapsed_times/b_cpg_cpg4_0.0.csv': {
		'condition': "CpG"
	},
	'./data/_processed/collapsed_times/aggre_1U_1.0.csv': {
		'condition': "1U IL-2"
	},
	'./data/_processed/collapsed_times/aggre_3U_2.0.csv': {
		'condition': "3U IL-2"
	},
	'./data/_processed/collapsed_times/aggre_10U_3.0.csv': {
		'condition': "10U IL-2"
	},
	'./data/_processed/collapsed_times/t_il2_20131218_1.0.csv': {
		'condition': "1U IL-2"
	},
	'./data/_processed/collapsed_times/t_il2_20131218_2.0.csv': {
		'condition': "3U IL-2"
	},
	'./data/_processed/collapsed_times/t_il2_20131218_3.0.csv': {
		'condition': "10U IL-2"
	},
	'./data/_processed/collapsed_times/t_il2_20140121_1.0.csv': {
		'condition': "1U IL-2"
	},
	'./data/_processed/collapsed_times/t_il2_20140121_2.0.csv': {
		'condition': "10U IL-2"
	},
	'./data/_processed/collapsed_times/t_il2_20140121_3.0.csv': {
		'condition': "3U IL-2"
	},
	'./data/_processed/collapsed_times/t_misc_20140211_1.0.csv': {
		'condition': "N4+CD28+IL-2"
	},
	'./data/_processed/collapsed_times/t_misc_20140211_2.0.csv': {
		'condition': "N4+CD28"
	},
	'./data/_processed/collapsed_times/t_misc_20140211_3.0.csv': {
		'condition': "N4"
	},
	'./data/_processed/collapsed_times/t_misc_20140211_4.0.csv': {
		'condition': "N4+IL-2"
	},
	'./data/_processed/collapsed_times/t_misc_20140325_1.0.csv': {
		'condition': "N4+IL-12"
	},
	'./data/_processed/collapsed_times/t_misc_20140325_2.0.csv': {
		'condition': "N4+CD28+IL-12"
	},
	'./data/_processed/collapsed_times/t_misc_20140325_3.0.csv': {
		'condition': "N4+CD28"
	},
	'./data/_processed/collapsed_times/t_misc_20140325_4.0.csv': {
		'condition': "N4"
	}
}

if __name__ == "__main__":
	### Import data
	loc_data = './data/_processed/collapsed_times'
	path_data = np.array([os.path.join(loc_data, file) for file in os.listdir(loc_data) if not file.startswith('.')])

	## SET ITERATION AND TUNING NUMBERS
	niter, ntune = 1000000, 10000
	# burn, thin = 100, 2
	nchain = ncore = 5
	nsubsample = 10000
	dt = 2000

	### HALF-NORMAL PRIORS FOR WEIBULL (UNIFORM PRIORS DOESN'T WORK FOR SOME REAONS)
	HALFN_STD = 200 * np.sqrt(1 / (1 - 2/np.pi))  # For sqrt(var(Half-Normal)) = 200
	for path in path_data:
		fname = os.path.basename(os.path.splitext(path)[0])
		print(f'\n======================== BEGIN {fname} ========================')

		df = pd.read_csv(path)
		max_time = max(df.max())
		xrange = xr.DataArray(np.linspace(0, max_time, dt), dims="x")

		lstys = ['-', '--', '-.', ':']
		colors = ['blue', 'orange', 'green', 'red']
		fig1, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
		# fig1.suptitle(f"[{fname}][{DATA_PROPERTY[path]['condition']}] Bayesian inference fit distribution")
		axes[0].set_xlim(left=0, right=max_time)
		axes[2].set_xlabel("Time (hour)")

		fig2, ax2 = plt.subplots(nrows=2, ncols=2)
		# fig2.suptitle(f"WAIC")
		ax2 = ax2.flat
		for i, (var, data) in enumerate(df.items()):
			print(f'------------------------ BEGIN {var} ({fname}) ------------------------')
			# Decide which axis to draw
			if var in ['tdiv0', 'tdiv']: ploc = 0; axes[0].set_title(f"Time to first division ({VAR_MAP['tdiv0']}) & Avg. Subsequent division time ({VAR_MAP['tdiv']})", x=0.01, ha='left')
			elif var == 'tld': ploc = 1; axes[1].set_title(f"Time to last division ({VAR_MAP['tld']})", x=0.01, ha='left')
			elif var == 'tdie': ploc = 2; axes[2].set_title(f"Time to death ({VAR_MAP['tdie']})", x=0.01, ha='left')
			axes[ploc].set_ylabel("eCDF")
			if i == 0: ax2[0].set_title(f"Time to first division ({VAR_MAP['tdiv0']})", fontdict={'color': 'blue'})
			if i == 1: ax2[1].set_title(f"Avg. Subsequent division time ({VAR_MAP['tdiv']})", fontdict={'color': 'orange'})
			if i == 2: ax2[2].set_title(f"Time to last division ({VAR_MAP['tld']})", fontdict={'color': 'green'})
			if i == 3: ax2[3].set_title(f"Time to death ({VAR_MAP['tdie']})", fontdict={'color': 'red'})

			data = data.dropna().to_numpy()
			if len(data) > 2:
				ECDF = ecdf(data)
				axes[ploc].step(ECDF[0], ECDF[1], color=colors[i], where='post', lw=1.5)
				axes[ploc].set_ylim(bottom=0, top=1)

				#########################################################################################################
				# 									      GAMMA DISTRIBUTION											#
				#########################################################################################################
				print("-->>>>>> GAMMA")
				with pm.Model() as gamma_model: 
					## Define uninformative priors
					## class pymc3.distributions.continuous.Uniform(lower=0, upper=1, *args, **kwargs)
					alpha1 = pm.Uniform('alpha1', lower=1E-10, upper=200)
					beta1 = pm.Uniform('beta1', lower=1E-10, upper=200)

					## class pymc3.distributions.continuous.Gamma(alpha=None, beta=None, mu=None, sigma=None, sd=None, *args, **kwargs)
					gamma = pm.Gamma('gamma', alpha=alpha1, beta=beta1, observed=data)

					gamma_trace = pm.sample(draws=niter, tune=ntune, chains=nchain, cores=ncore, target_accept=0.95, init='jitter+adapt_diag', return_inferencedata=True)
					# gamma_trace = pm.sample(draws=niter, tune=ntune, chains=nchain, cores=ncore, target_accept=0.95, init='jitter+adapt_diag')  # For trace plot
				# gamma_trace = gamma_trace[burn::thin]
				# plot_traces(gamma_trace)
				gamma_waic = pm.waic(gamma_trace, gamma_model, scale='deviance')  # calculate WAIC
				gamma_summary = pm.summary(gamma_trace)

				## Sample from posterior distribution to generate cdfs and plot over eCDF
				# https://stackoverflow.com/questions/63608116/plot-fit-of-gamma-distribution-with-pymc3
				# all_gamma_cdfs = xr.apply_ufunc(
				# 	lambda alpha, beta, x: sps.gamma(a=alpha, scale=1/beta).cdf(x),
				# 	gamma_trace.posterior["alpha1"], gamma_trace.posterior["beta1"], xrange
				# )
				## Get random subset of the posterior
				idx = rng.choice(gamma_trace.posterior.alpha1.size, nsubsample)
				post = gamma_trace.posterior.stack(sample=("chain", "draw")).isel(sample=idx)
				gamma_cdfs = xr.apply_ufunc(
					lambda alpha, beta, x: sps.gamma(a=alpha, scale=1/beta).cdf(x),
					post["alpha1"], post["beta1"], xrange
				)
				## Plot results, for proper plotting, "x" dim must be the first
				# axes[0].plot(xrange, cdfs.transpose("x", ...), ls='-', lw=1, alpha=0.2)
				gamma_mean, gamma_low, gamma_high = confidence_band(gamma_cdfs)
				axes[ploc].plot(xrange, gamma_mean, ls=lstys[0], color=colors[i], label=f"Gam({gamma_summary.at['alpha1', 'mean']:.2f}, {1/gamma_summary.at['beta1', 'mean']:.2f})")
				axes[ploc].plot(xrange, gamma_low, ls=lstys[0], lw=0.3, color=colors[i])
				axes[ploc].plot(xrange, gamma_high, ls=lstys[0], lw=0.3, color=colors[i])
				axes[ploc].fill_between(xrange, gamma_low, gamma_high, ls=lstys[0], color=colors[i], ec=(0, 0, 0, 0), alpha=0.1)
				print("> Result GAMMA:")
				print(gamma_summary, end='\n\n')

				#########################################################################################################
				# 									     LOGNORMAL DISTRIBUTION											#
				#########################################################################################################
				print("-->>>>>> LOGNORMAL")
				with pm.Model() as lnorm_model: 
					## Define uninformative priors
					## class pymc3.distributions.continuous.Uniform(lower=0, upper=1, *args, **kwargs)
					mu1 = pm.Uniform('mu1', lower=1E-10, upper=200)
					sigma1 = pm.Uniform('sigma1', lower=1E-10, upper=200)

					## class pymc3.distributions.continuous.Lognormal(mu=0, sigma=None, tau=None, sd=None, *args, **kwargs)
					lnorm = pm.Lognormal('lnorm', mu=mu1, sigma=sigma1, observed=data)  # For trace plot

					lnorm_trace = pm.sample(draws=niter, tune=ntune, chains=nchain, cores=ncore, target_accept=0.95, init='jitter+adapt_diag', return_inferencedata=True)
					# lnorm_trace = pm.sample(draws=niter, tune=ntune, chains=nchain, cores=ncore, target_accept=0.95, init='jitter+adapt_diag')  # For trace plot
				# lnorm_trace = lnorm_trace[burn::thin]
				# plot_traces(lnorm_trace)
				lnorm_waic = pm.waic(lnorm_trace, lnorm_model, scale='deviance')
				lnorm_summary = pm.summary(lnorm_trace)
				
				## Get random subset of the posterior
				idx = rng.choice(lnorm_trace.posterior.mu1.size, nsubsample)
				post = lnorm_trace.posterior.stack(sample=("chain", "draw")).isel(sample=idx)
				lnorm_cdfs = xr.apply_ufunc(
					lambda mu, sigma, x: sps.lognorm(s=sigma, scale=np.exp(mu)).cdf(x),
					post["mu1"], post["sigma1"], xrange
				)
				lnorm_mean, lnorm_low, lnorm_high = confidence_band(lnorm_cdfs)
				axes[ploc].plot(xrange, lnorm_mean, ls=lstys[1], color=colors[i], label=f"LN({np.exp(lnorm_summary.at['mu1', 'mean']):.2f}, {lnorm_summary.at['sigma1', 'mean']:.2f})")
				axes[ploc].plot(xrange, lnorm_low, ls=lstys[1], lw=0.3, color=colors[i])
				axes[ploc].plot(xrange, lnorm_high, ls=lstys[1], lw=0.3, color=colors[i])
				axes[ploc].fill_between(xrange, lnorm_low, lnorm_high, ls=lstys[1], color=colors[i], ec=(1, 1, 1, 1), alpha=0.1)
				print("> Result LOGNORMAL:")
				print(lnorm_summary, end='\n\n')

				#########################################################################################################
				# 									     NORMAL DISTRIBUTION											#
				#########################################################################################################
				print("-->>>>>> NORMAL")
				with pm.Model() as norm_model: 
					## Define uninformative priors
					## class pymc3.distributions.continuous.Uniform(lower=0, upper=1, *args, **kwargs)
					mu2 = pm.Uniform('mu2', lower=1E-10, upper=200)
					sigma2 = pm.Uniform('sigma2', lower=1E-10, upper=200)

					norm = pm.Normal('norm', mu=mu2, sigma=sigma2, observed=data)

					norm_trace = pm.sample(draws=niter, tune=ntune, chains=nchain, cores=ncore, target_accept=0.95, init='jitter+adapt_diag', return_inferencedata=True)
					# norm_trace = pm.sample(draws=niter, tune=ntune, chains=nchain, cores=ncore, target_accept=0.95, init='jitter+adapt_diag')  # For trace plot
				# norm_trace = norm_trace[burn::thin]
				# plot_traces(norm_trace)
				norm_waic = pm.waic(norm_trace, norm_model, scale='deviance')
				norm_summary = pm.summary(norm_trace)

				## Get random subset of the posterior
				idx = rng.choice(norm_trace.posterior.mu2.size, nsubsample)
				post = norm_trace.posterior.stack(sample=("chain", "draw")).isel(sample=idx)
				norm_cdfs = xr.apply_ufunc(
					lambda mu, sigma, x: sps.norm(loc=mu, scale=sigma).cdf(x),
					post["mu2"], post["sigma2"], xrange
				)
				norm_mean, norm_low, norm_high = confidence_band(norm_cdfs)
				axes[ploc].plot(xrange, norm_mean, ls=lstys[2], color=colors[i], label=f"N({norm_summary.at['mu2', 'mean']:.2f}, {norm_summary.at['sigma2', 'mean']:.2f})")
				axes[ploc].plot(xrange, norm_low, ls=lstys[2], lw=0.3, color=colors[i])
				axes[ploc].plot(xrange, norm_high, ls=lstys[2], lw=0.3, color=colors[i])
				axes[ploc].fill_between(xrange, norm_low, norm_high, ls=lstys[2], color=colors[i], ec=(1, 1, 1, 1), alpha=0.1)
				print("> Resul NORMAL:")
				print(norm_summary, end='\n\n')

				#########################################################################################################
				# 									     WEIBULL DISTRIBUTION											#
				#########################################################################################################
				print("-->>>>>> WEIBULL")
				with pm.Model() as weibull_model: 
					## Define uninformative priors: Weibull seems more sensitive to the priors than the other three distribution classes (Uniform fails)
					## class pymc3.distributions.continuous.Uniform(lower=0, upper=1, *args, **kwargs)
					# alpha2 = pm.Uniform('alpha2', lower=1E-10, upper=200)
					# beta2 = pm.Uniform('beta2', lower=1E-10, upper=200)

					## class pymc3.distributions.continuous.HalfNormal(sigma=None, tau=None, sd=None, *args, **kwargs)
					alpha2 = pm.HalfNormal('alpha2', sigma=HALFN_STD)
					beta2 = pm.HalfNormal('beta2', sigma=HALFN_STD)

					## class pymc3.distributions.continuous.Weibull(alpha, beta, *args, **kwargs)
					weibull = pm.Weibull('weibull', alpha=alpha2, beta=beta2, observed=data)

					weibull_trace = pm.sample(draws=niter, tune=ntune, chains=nchain, cores=ncore, target_accept=0.95, init='jitter+adapt_diag', return_inferencedata=True)
					# weibull_trace = pm.sample(draws=niter, tune=ntune, chains=nchain, cores=ncore, target_accept=0.95, init='jitter+adapt_diag')  # For trace plot
				# weibull_trace = weibull_trace[burn::thin]
				# plot_traces(weibull_trace)
				weibull_waic = pm.waic(weibull_trace, weibull_model, scale='deviance')
				weibull_summary = pm.summary(weibull_trace)

				## Get random subset of the posterior
				idx = rng.choice(weibull_trace.posterior.alpha2.size, nsubsample)
				post = weibull_trace.posterior.stack(sample=("chain", "draw")).isel(sample=idx)
				weibull_cdfs = xr.apply_ufunc(
					lambda alpha, beta, x: sps.weibull_min(c=alpha, scale=beta).cdf(x),
					post["alpha2"], post["beta2"], xrange
				)
				weibull_mean, weibull_low, weibull_high = confidence_band(weibull_cdfs)
				axes[ploc].plot(xrange, weibull_mean, ls=lstys[3], color=colors[i], label=f"Wei({weibull_summary.at['alpha2', 'mean']:.2f}, {weibull_summary.at['beta2', 'mean']:.2f})")
				axes[ploc].plot(xrange, weibull_low, ls=lstys[3], lw=0.3, color=colors[i])
				axes[ploc].plot(xrange, weibull_high, ls=lstys[3], lw=0.3, color=colors[i])
				axes[ploc].fill_between(xrange, weibull_low, weibull_high, ls=lstys[3], color=colors[i], ec=(1, 1, 1, 1), alpha=0.1)
				print("> Resul WEIBULL:")
				print(weibull_summary, end='\n\n')

				axes[ploc].grid(True, which='major', axis='both', linestyle='--')
				if ploc == 0  :
					axes[ploc].legend(loc='lower right', fontsize=10, ncol=2, columnspacing=0.2)
				else:
					axes[ploc].legend(loc='upper left', fontsize=10, ncol=1)


				#########################################################################################################
				# 								COMPARE DISTRIBUTIONS (WAIC & LOO)										#
				#########################################################################################################
				print("-->>>>>> MODEL SELECTION: WAIC")
				## https://arviz-devs.github.io/arviz/generated/arviz.compare.html#arviz.compare
				dfwaic = pm.compare(
					{'Gamma': gamma_trace, 'Log-normal': lnorm_trace, 'Normal': norm_trace, 'Weibull': weibull_trace}, 
					ic='WAIC', method='BB-pseudo-BMA', b_samples=1000000, alpha=1, seed=None, scale='deviance')
				print(dfwaic, end='\n\n')
				## https://arviz-devs.github.io/arviz/generated/arviz.plot_compare.html#arviz.plot_compare
				pm.compareplot(dfwaic, ax=ax2[i])

				# print("-->>>>>> MODEL SELECTION: Leave-One-Out (LOO) Cross-Validation")
				# dfloo = pm.compare(
				# 	{'Gamma': gamma_trace, 'Log-normal': lnorm_trace, 'Normal': norm_trace, 'Weibull': weibull_trace},
				# 	ic='LOO', method='BB-pseudo-BMA', b_samples=1000000, alpha=1, seed=None, scale='deviance')
				# print(dfloo)
			else:
				print(f'CANNOT PROCEED. {var} HAS NO DATA!')
				ax2[i].annotate("NA", xy=(0.5, 0.5), xycoords='axes fraction', weight='bold', fontsize=14)

			print(f'------------------------ END of {var} ({fname}) ------------------------\n\n')
		print(f'======================== END of {fname} ========================')
		ax2[0].set_xlabel("")
		ax2[1].set_xlabel("")
		fig1.tight_layout(rect=(0, 0, 1, 1))
		fig1.subplots_adjust(hspace=0.17, wspace=0)
		fig2.tight_layout(rect=(0, 0, 1, 1))
		fig2.subplots_adjust(hspace=0.2, wspace=0.2)

		fig1.savefig(f'./out/Fig4-dist/{fname}_f1.pdf', dpi=300)
		fig2.savefig(f'./out/Fig4-dist/{fname}_f2.pdf', dpi=300)
