"""
Last update: 2-September-2021

Bayesian inference to select the best distribution class (i.e. model selection) given the observed times
[Output] Fig4 in the main article; FigS4, FigS5 in the Supplementary Material
"""
import sys, os
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 999999)
pd.set_option('display.max_columns', 999999)
pd.set_option('display.expand_frame_repr', False)
import seaborn as sns
import matplotlib as mpl; mpl.use('pdf')
import matplotlib.pyplot as plt
import scipy.stats as sps
import arviz as az
import pymc3 as pm
import theano as TT
import xarray as xr
from _func import ecdf
rng = np.random.RandomState(seed=10230721)

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

VAR_MAP = {'tdiv0': "$T_{div}^0$", 'tdiv': "$T_{div}^{k\geq1}$", 'tld': "$T_{ld}$", 'tdie': "$T_{die}$"}
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


class DelayedLognormal(pm.Continuous):
	def __init__(self, mu=0.0, tau=None, sigma=None, c=0.0, *args, **kwargs):
		super(DelayedLognormal, self).__init__(*args, **kwargs)
		self.mu = mu
		self.c = c
		self.tau, self.sigma = pm.distributions.continuous.get_tau_sigma(tau=tau, sigma=sigma)
	
	def logp(self, value):
		mu = self.mu
		tau = self.tau
		x = value - self.c

		bound = pm.distributions.continuous.bound
		return bound(-0.5*tau*(pm.math.log(x) - mu)**2 + \
					 0.5*pm.math.log(tau/(2.*np.pi)) - \
					 pm.math.log(x),
					 tau > 0, x > 0)

class DelayedExponential(pm.Continuous):
	def __init__(self, lam=None, c=0.0, *args, **kwargs):
		super(DelayedExponential, self).__init__(*args, **kwargs)
		self.lam = lam
		self.c = c
	
	def logp(self, value):
		lam = self.lam
		x = value - self.c

		bound = pm.distributions.continuous.bound
		return bound(pm.math.log(lam) - lam * x, lam > 0, x > 0)
	
def get_dexp_cdf_from_posterior(x, lamb, c):
	s = []
	for l, t in zip(lamb.values, c.values):
		s.append([1. - np.exp(-l * (xx - t)) if xx >= t else 0. for xx in x.values.ravel()])
	return np.array(s)

def confidence_band(x):
	mean = np.mean(x, axis=0)
	low, high = np.percentile(x, q=[2.5, 97.5], interpolation='nearest', axis=0)
	return mean, low, high

def _waic(trace, name, idx=0):
	log_likelihood = trace.log_likelihood[list(trace.log_likelihood.data_vars)[0]]
	log_likelihood = log_likelihood.stack(sample=("chain", "draw"))  # if has more than 1 chain

	shape = log_likelihood.shape
	n_samples = shape[-1]  # samples from posterior distribution
	n_data = np.product(shape[:-1])  # number of data points

	# Eq.2 (lppd_i = lppd per data point): log( 1/S * \sum_s^S( P(z_i|Theta_s) ) = logsumexp(logLike) - log(S)
	lppd_i = np.log(np.sum(np.exp(log_likelihood.values), axis=1)) - np.log(n_samples)  
	vars_lpd = log_likelihood.var(dim="sample").values
	waic_i = -2. * (lppd_i - vars_lpd)

	waic = np.sum(waic_i)
	p_waic = np.sum(vars_lpd)  # penalty term
	
	# Standard error of WAIC
	se = np.sqrt(n_data * np.var(waic_i))

	res = pd.DataFrame({'name': name, 'waic': waic, 'p_waic': p_waic, 'se': se, 'waic_i': [waic_i]}, index=pd.Index([idx]))
	return res

def compare_waics(traces, names):
	waics = []
	for idx, (trace, name) in enumerate(zip(traces, names)):
		waic = _waic(trace, name, idx)
		waics.append(waic)
	waics = pd.concat(waics)
	waics = waics.sort_values(by='waic').reset_index(drop=True)

	# difference of waic w.r.t the best model
	waics['delta_waic'] = waics.loc[:,'waic'] - waics.at[0,'waic']

	# standard error of difference of waic w.r.t the best model
	n_data = len(waics.at[0,'waic_i'])
	waics['delta_se'] = np.sqrt(
		n_data * np.var(np.vstack(waics.loc[:,'waic_i'].values) - waics.at[0,'waic_i'][np.newaxis,:], axis=1)
	)
	waics.drop(columns='waic_i', inplace=True)
	waics.index.name = 'rank'
	return waics

if __name__ == "__main__":
	### Import data
	loc_data = './data/_processed/collapsed_times'
	path_data = np.array([os.path.join(loc_data, file) for file in os.listdir(loc_data) if not file.startswith('.')])

	excl = ['./data/_processed/collapsed_times/t_il2_20131218_1.0.csv',
			'./data/_processed/collapsed_times/t_il2_20131218_2.0.csv',
			'./data/_processed/collapsed_times/t_il2_20131218_3.0.csv',
			'./data/_processed/collapsed_times/t_il2_20140121_1.0.csv',
			'./data/_processed/collapsed_times/t_il2_20140121_2.0.csv',
			'./data/_processed/collapsed_times/t_il2_20140121_3.0.csv']
	path_data = [path for path in path_data if path not in excl]  # exclude individual IL-2 dataset (use aggregated one)

	## SET ITERATION AND TUNING NUMBERS
	niter, ntune = 1000000, 10000
	nchain = ncore = 5
	nsubsample = 10000
	dt = 1000

	### HALF-NORMAL PRIORS FOR WEIBULL (UNIFORM PRIORS DOESN'T WORK)
	HALFN_STD = 500 * np.sqrt(1 / (1 - 2/np.pi))
	for path in path_data:
		fname = os.path.basename(os.path.splitext(path)[0])
		print(f'\n======================== BEGIN {fname} ========================')

		df = pd.read_csv(path)
		max_time = max(df.max())
		xrange = xr.DataArray(np.linspace(0, max_time, dt), dims="x")

		lstys = ['-', '--', '-.', ':', (0, (3, 5, 1, 5, 1, 5)), (0, (3, 1, 1, 1, 1, 1))]
		colors = ['blue', 'orange', 'green', 'red']
		fig1, axes = plt.subplots(nrows=4, ncols=1, sharex=True)
		axes[0].set_xlim(left=0, right=max_time)
		axes[3].set_xlabel("Time (hour)")

		fig2, ax2 = plt.subplots(nrows=2, ncols=2)
		ax2 = ax2.flat
		for i, (var, data) in enumerate(df.items()):
			print(f'------------------------ BEGIN {var} ({fname}) ------------------------')
			# Decide which axis to draw
			if var  == 'tdiv0': 
				ploc = 0
				axes[0].set_title(f"Time to first division ({VAR_MAP['tdiv0']})", x=0.01, ha='left', fontsize=13.5)
			elif var == 'tdiv':
				ploc = 1
				axes[1].set_title(f"Avg. sub div time ({VAR_MAP['tdiv']})", x=0.01, ha='left', fontsize=13.5)
			elif var == 'tld':
				ploc = 2
				axes[2].set_title(f"Time to last division ({VAR_MAP['tld']})", x=0.01, ha='left', fontsize=13.5)
			elif var == 'tdie': 
				ploc = 3
				axes[3].set_title(f"Time to death ({VAR_MAP['tdie']})", x=0.01, ha='left', fontsize=13.5)
			axes[ploc].set_ylabel("eCDF")

			if i == 0: ax2[0].set_title(f"Time to first division ({VAR_MAP['tdiv0']})", fontdict={'color': 'blue'})
			if i == 1: ax2[1].set_title(f"Avg. sub div time ({VAR_MAP['tdiv']})", fontdict={'color': 'orange'})
			if i == 2: ax2[2].set_title(f"Time to last division ({VAR_MAP['tld']})", fontdict={'color': 'green'})
			if i == 3: ax2[3].set_title(f"Time to death ({VAR_MAP['tdie']})", fontdict={'color': 'red'})

			data = data.dropna().to_numpy()
			if len(data) > 1:
				ECDF = ecdf(data)
				axes[ploc].step(ECDF[0], ECDF[1], color=colors[i], where='post', lw=1.5)
				axes[ploc].set_ylim(bottom=0, top=1)

				#########################################################################################################
				# 									      GAMMA DISTRIBUTION											#
				#########################################################################################################
				print("-->>>>>> GAMMA")
				with pm.Model() as gamma_model: 
					alpha1 = pm.Uniform('alpha1', lower=1E-10, upper=200)
					beta1 = pm.Uniform('beta1', lower=1E-10, upper=200)
					gamma = pm.Gamma('gamma', alpha=alpha1, beta=beta1, observed=data)
					gamma_trace = pm.sample(draws=niter,
											tune=ntune, 
											chains=nchain, 
											cores=ncore,
											random_seed=17198734,
											target_accept=0.95, 
											init='jitter+adapt_diag', 
											return_inferencedata=True)
				gamma_waic = az.waic(gamma_trace, gamma_model, scale='deviance')  # calculate WAIC
				gamma_summary = pm.summary(gamma_trace)

				## Sample from posterior distribution to generate cdfs and plot over eCDF
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
					mu1 = pm.Uniform('mu1', lower=1E-10, upper=200)
					sigma1 = pm.Uniform('sigma1', lower=1E-10, upper=200)
					lnorm = pm.Lognormal('lnorm', mu=mu1, sigma=sigma1, observed=data)  # For trace plot
					lnorm_trace = pm.sample(draws=niter, 
											tune=ntune, 
											chains=nchain, 
											cores=ncore, 
											random_seed=33541166,
											target_accept=0.95,
											init='jitter+adapt_diag', 
											return_inferencedata=True)
				lnorm_waic = az.waic(lnorm_trace, lnorm_model, scale='deviance')
				lnorm_summary = pm.summary(lnorm_trace)
				
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
					mu2 = pm.Uniform('mu2', lower=1E-10, upper=200)
					sigma2 = pm.Uniform('sigma2', lower=1E-10, upper=200)
					norm = pm.Normal('norm', mu=mu2, sigma=sigma2, observed=data)
					norm_trace = pm.sample(draws=niter, 
											tune=ntune, 
											chains=nchain, 
											cores=ncore, 
											random_seed=67754922,
											target_accept=0.95, 
											init='jitter+adapt_diag', 
											return_inferencedata=True)
				norm_waic = az.waic(norm_trace, norm_model, scale='deviance')
				norm_summary = pm.summary(norm_trace)

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
				print("> Result NORMAL:")
				print(norm_summary, end='\n\n')

				#########################################################################################################
				# 									     WEIBULL DISTRIBUTION											#
				#########################################################################################################
				print("-->>>>>> WEIBULL")
				with pm.Model() as weibull_model: 
					# alpha2 = pm.Uniform('alpha2', lower=1E-10, upper=200)
					# beta2 = pm.Uniform('beta2', lower=1E-10, upper=200)
					alpha2 = pm.HalfNormal('alpha2', sigma=HALFN_STD)
					beta2 = pm.HalfNormal('beta2', sigma=HALFN_STD)
					weibull = pm.Weibull('weibull', alpha=alpha2, beta=beta2, observed=data)
					weibull_trace = pm.sample(draws=niter, 
												tune=ntune, 
												chains=nchain, 
												cores=ncore,
												random_seed=26914075,
												target_accept=0.95, 
												init='jitter+adapt_diag', 
												return_inferencedata=True)
				weibull_waic = az.waic(weibull_trace, weibull_model, scale='deviance')
				weibull_summary = pm.summary(weibull_trace)

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
				print("> Result WEIBULL:")
				print(weibull_summary, end='\n\n')

				#########################################################################################################
				# 									   DELAYED EXP DISTRIBUTION											#
				#########################################################################################################
				print("-->>>>>> DELAYED EXPONENTIAL")
				with pm.Model() as dexp_model: 
					lamb = pm.Uniform('lamb', lower=1E-10, upper=2.)
					c = pm.Uniform('c', lower=0., upper=np.nanmin(data))

					dexp = DelayedExponential('dexp', lam=lamb, c=c, observed=data)
					dexp_trace = pm.sample(draws=niter,
										   tune=ntune,
										   chains=nchain,
										   cores=ncore,
										   random_seed=89475927,
										   target_accept=0.95,
										   init='jitter+adapt_diag',
										   return_inferencedata=True)
				dexp_waic = az.waic(dexp_trace, dexp_model, scale='deviance')
				dexp_summary = pm.summary(dexp_trace)

				idx = rng.choice(dexp_trace.posterior.lamb.size, nsubsample)
				post = dexp_trace.posterior.stack(sample=("chain", "draw")).isel(sample=idx)
				dexp_cdfs = get_dexp_cdf_from_posterior(xrange, post["lamb"], post["c"])
				dexp_mean, dexp_low, dexp_high = confidence_band(dexp_cdfs)
				axes[ploc].plot(xrange, dexp_mean, ls=lstys[4], color=colors[i], label=rf"dExp({dexp_summary.at['lamb', 'mean']:.2f}, {dexp_summary.at['c', 'mean']:.2f})")
				axes[ploc].plot(xrange, dexp_low, ls=lstys[4], lw=0.3, color=colors[i])
				axes[ploc].plot(xrange, dexp_high, ls=lstys[4], lw=0.3, color=colors[i])
				axes[ploc].fill_between(xrange, dexp_low, dexp_high, ls=lstys[4], color=colors[i], ec=(0, 0, 0, 0), alpha=0.1)
				print("> Result DELAYED EXPONENTIAL:")
				print(dexp_summary, end='\n\n')

				#########################################################################################################
				# 									DELAYED LOGNORMAL DISTRIBUTION										#
				#########################################################################################################
				print("-->>>>>> DELAYED LOGNORMAL")
				with pm.Model() as dLN_model: 
					muLN = pm.Uniform('muLN', lower=1E-10, upper=200)
					sigmaLN = pm.Uniform('sigmaLN', lower=1E-10, upper=200)
					c = pm.Uniform('c', lower=0., upper=np.nanmin(data))
					
					dLN = DelayedLognormal('dLN', mu=muLN, sigma=sigmaLN, c=c, observed=data)
					dLN_trace = pm.sample(draws=niter,
										  tune=ntune,
										  chains=nchain,
										  cores=ncore,
										  random_seed=55150623,
									   	  target_accept=0.95,
										  init='jitter+adapt_diag',
										  return_inferencedata=True)
				dLN_waic = az.waic(dLN_trace, dLN_model, scale='deviance')
				dLN_summary = pm.summary(dLN_trace)

				idx = rng.choice(dLN_trace.posterior.muLN.size, nsubsample)
				post = dLN_trace.posterior.stack(sample=("chain", "draw")).isel(sample=idx)
				dLN_cdfs = xr.apply_ufunc(
					lambda mu, sigma, c, x: sps.lognorm(s=sigma, scale=np.exp(mu), loc=c).cdf(x),
					post["muLN"], post["sigmaLN"], post["c"], xrange
				)
				dLN_mean, dLN_low, dLN_high = confidence_band(dLN_cdfs)
				axes[ploc].plot(xrange, dLN_mean, ls=lstys[5], color=colors[i], 
								label=rf"dLN({dLN_summary.at['muLN', 'mean']:.2f}, {dLN_summary.at['sigmaLN', 'mean']:.2f}, {dLN_summary.at['c', 'mean']:.2f})")
				axes[ploc].plot(xrange, dLN_low, ls=lstys[5], lw=0.3, color=colors[i])
				axes[ploc].plot(xrange, dLN_high, ls=lstys[5], lw=0.3, color=colors[i])
				axes[ploc].fill_between(xrange, dLN_low, dLN_high, ls=lstys[5], color=colors[i], ec=(0, 0, 0, 0), alpha=0.1)
				print("> Result DELAYED LOGNORAML:")
				print(dLN_summary, end='\n\n')



				axes[ploc].grid(True, which='major', axis='both', linestyle='--')
				axes[ploc].legend(fontsize=9.5, ncol=2, columnspacing=0.5)
				# if ploc == 0:
				# 	axes[ploc].legend(loc='lower right', fontsize=9.5, ncol=2, columnspacing=0.2)
				# else:
				# 	axes[ploc].legend(loc='upper left', fontsize=9.5, ncol=1)

				#########################################################################################################
				# 								COMPARE DISTRIBUTIONS (WAIC)											#
				#########################################################################################################
				print("-->>>>>> MODEL SELECTION: WAIC")
				my_waic = compare_waics([gamma_trace, lnorm_trace, norm_trace, weibull_trace, dexp_trace, dLN_trace], 
										['Gam', 'LN', 'N', 'Wei', 'dExp', 'dLN'])
				print(">>> Compare WAIC")
				print(my_waic, end='\n')

				dfwaic = az.compare(  # Identical to compare_waic() defined above. But it's convenient to use this for plotting the results
					{'Gam': gamma_trace, 'LN': lnorm_trace, 'N': norm_trace, 'Wei': weibull_trace, 'dExp': dexp_trace, 'dLN': dLN_trace}, 
					ic='WAIC', method='stacking', scale='deviance')
				print(">>> PyMC3/Arviz")
				print(dfwaic, end='\n\n')
				plt.rcParams.update({'axes.titlesize': 12, 'axes.labelsize': 12,
									'xtick.labelsize': 12, 'ytick.labelsize': 12})
				az.plot_compare(dfwaic, textsize=16, ax=ax2[i])
				plt.rcParams.update({'axes.titlesize': rc['axes.titlesize'], 'axes.labelsize': rc['axes.labelsize'],
									'xtick.labelsize': rc['xtick.labelsize'], 'ytick.labelsize': rc['ytick.labelsize']})
			else:
				print(f'CANNOT PROCEED. {var} HAS NO DATA!')
				axes[ploc].annotate("NA", xy=(0.45, 0.5), xycoords='axes fraction', weight='bold', fontsize=16)
				ax2[i].annotate("NA", xy=(0.45, 0.5), xycoords='axes fraction', weight='bold', fontsize=16)
				ax2[i].set_xticklabels([])
				ax2[i].set_yticklabels([])

			print(f'------------------------ END of {var} ({fname}) ------------------------\n\n')
		print(f'======================== END of {fname} ========================')
		ax2[0].set_xlabel("")
		ax2[1].set_xlabel("")
		fig1.tight_layout(rect=(0, 0, 1, 1))
		fig1.subplots_adjust(hspace=0.21, wspace=0)
		fig2.tight_layout(rect=(0, 0, 1, 1))
		fig2.subplots_adjust(hspace=0.255, wspace=0.2)

		fig1.savefig(f'./out/Fig4-dist/{fname}_f1.pdf', dpi=300)
		fig2.savefig(f'./out/Fig4-dist/{fname}_f2.pdf', dpi=300)
