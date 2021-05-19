"""
Last edit: 16-May-2021

Scenario 1 of application of the Cyton2 model fitting CpG-stimulated B cell FACS data
[Output] Data files (in excel format) for Fig5A1-3, Fig5B and FigS6 -> Run "Fig5-Plot.py" for generating the plots
"""
import sys, os, time, datetime
import tqdm
import numpy as np
import pandas as pd
import matplotlib as mpl; mpl.use('pdf')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
import lmfit as lmf
from src.parse import parse_data
from src.utils import conf_iterval, norm_pdf, norm_cdf, lognorm_pdf, lognorm_cdf
from src.model import Cyton15Model   # Full Cyton model. (Options: choose all variable to be either logN or N)
rng = np.random.RandomState(seed=89907530)

## Check library version
try:
	assert(sns.__version__=='0.10.1')
except AssertionError as ae:
	print("[VersionError] Please check if the following version of seaborn library is installed:")
	print("seaborn==0.10.1")
	sys.exit()

rc = {
	'figure.figsize': (9, 7),
	# 'font.size': 14, 'axes.titlesize': 14, 'axes.labelsize': 12,
	# 'xtick.labelsize': 14, 'ytick.labelsize': 14,
	# 'legend.fontsize': 14, 'legend.title_fontsize': None,
	# 'axes.grid': True, 'axes.grid.axis': 'x', 'axes.grid.axis': 'y',
	'axes.axisbelow': True, 'axes.titlepad': 0,
	'axes.spines.top': False, 'axes.spines.right': False,
	'axes.spines.left': True, 'axes.spines.bottom': True,
	'ytick.left': True, 'xtick.bottom': True,
	'lines.markersize': 7.5, 'lines.linewidth': 1,
	'errorbar.capsize': 2.5
}
sns.set(style='white', rc=rc)

DT = 0.5            # [Cyton Model] Time step
ITER_SEARCH = 100   # [Cyton Model] Number of initial search
MAX_NFEV = None     # [LMFIT] Maximum number of function evaluation
LM_FIT_KWS = {      # [LMFIT/SciPy] Key-word arguements pass to LMFIT minimizer for Levenberg-Marquardt algorithm
	# 'ftol': 1E-10,  # Relative error desired in the sum of squares. DEFAULT: 1.49012E-8
	# 'xtol': 1E-10,  # Relative error desired in the approximate solution. DEFAULT: 1.49012E-8
	# 'gtol': 0.0,    # Orthogonality desired between the function vector and the columns of the Jacobian. DEFAULT: 0.0
	'epsfcn': 1E-4  # A variable used in determining a suitable step length for the forward-difference approximation of the Jacobian (for Dfun=None). Normally the actual step length will be sqrt(epsfcn)*x If epsfcn is less than the machine precision, it is assumed that the relative errors are of the order of the machine precision. Default value is around 2E-16. As it turns out, the optimisation routine starts by making a very small move (functinal evaluation) and calculating the finite-difference Jacobian matrix to determine the direction to move. The default value is too small to detect sensitivity of 'm' parameter.
}

LOGNORM = True	 	# All variables ~ LN(m,s); Otherwisee, all variables ~ N(mu,sig)
RGS = 95            # [BOOTSTRAP] alpha for confidence interval
ITER_BOOTS = 1000   # [BOOTSTRAP] Bootstrap samples

### CYTON MODEL: OBJECTIVE FUNCTION
def residual(pars, x, data=None, model=None):
	vals = pars.valuesdict()
	mUns, sUns = vals['mUns'], vals['sUns']
	mDiv0, sDiv0 = vals['mDiv0'], vals['sDiv0']
	mDD, sDD = vals['mDD'], vals['sDD']
	mDie, sDie = vals['mDie'], vals['sDie']
	m, p = vals['m'], vals['p']

	pred = model.evaluate(mUns, sUns, mDiv0, sDiv0, mDD, sDD, mDie, sDie, m, p)

	return (data - pred)

def subsample(df, n):
	sampled_series = []
	for _, row in df.iterrows():
		sampled_series.append(row.sample(n=n, replace=True, random_state=rng))

	new_df = {}
	for rep in range(n):
		new_df[rep] = np.array(sampled_series)[:,rep]

	return pd.DataFrame(new_df, index=df.index)

def fit_reps(inputs):
	key, df, reader, sl, lognorm = inputs
	icnd = 0

	hts = reader.harvested_times[icnd]
	mgen = reader.generation_per_condition[icnd]
	condition = reader.condition_names[icnd]

	### ANALYSIS: ESTIMATE DIVISION PARAMETERS BY FITTING CYTON MODEL
	pars = {  # Initial values
			'mUns': 1000, 'sUns': 1E-3,  	# Unstimulated death time (NOT USED HERE)
			'mDiv0': 30, 'sDiv0': 0.2,      # Time to first division
			'mDD': 60, 'sDD': 0.3,    		# Time to division destiny
			'mDie': 80, 'sDie': 0.4,		# Time to death
			'm': 10, 'p': 1					# Subseqeunt division time & Activation probability (ASSUME ALL CELLS ACTIVATED)
		}
	bounds = {
		'lb': {  # Lower bounds
			'mUns': 1E-4, 'sUns': 1E-3,
			'mDiv0': 1E-2, 'sDiv0': 1E-2,
			'mDD': 1E-2, 'sDD': 1E-2,
			'mDie': 1E-2, 'sDie': 1E-2,
			'm': 5, 'p': 0
		},
		'ub': {  # Upper bounds
			'mUns': 1000, 'sUns': 2,
			'mDiv0': 500, 'sDiv0': 2,
			'mDD': 500, 'sDD': 2,
			'mDie': 500, 'sDie': 2,
			'm': 50, 'p': 1
		}
	}
	if lognorm: pass
	else:
		pars['mDiv0'], pars['sDiv0'] = 30, 10
		pars['mDD'], pars['sDD'] = 60, 10
		pars['mDie'], pars['sDie'] = 80, 10
		bounds['lb']['mDiv0'] = 0
		bounds['ub']['mDiv0'], bounds['ub']['sDiv0'] = 200, 200
		bounds['lb']['mDD'] = 0
		bounds['ub']['mDD'], bounds['ub']['sDD'] = 200, 200
		bounds['lb']['mDie'] = 0
		bounds['ub']['mDie'], bounds['ub']['sDie'] = 200, 200
	vary = {  # True = Subject to change; False = Lock parameter
		'mUns': False, 'sUns': False,
		'mDiv0': True, 'sDiv0': True,
		'mDD': True, 'sDD': True,
		'mDie': True, 'sDie': True,
		'm': True, 'p': False
	}

	boots = {  # This entire process is similar to bootstrapping!
		'mUns': [], 'sUns': [],
		'mDiv0': [], 'sDiv0': [],
		'mDD': [], 'sDD': [],
		'mDie': [], 'sDie': [],
		'm': [], 'p': [], 'N0': [],
		'algo': [], 'mse-in': [], 'mse-out': [], 'rmse-in': [], 'rmse-out': []
	}

	### PREPARE DATA
	conv_df = pd.DataFrame(df['cgens']['rep'][icnd])
	conv_df.index = hts

	nreps = []
	for idx, row in conv_df.iterrows():
		nreps.append(len(row.dropna()))
	data = df['cgens']['rep'][icnd]  # n(g,t): number of cells in generation g at time t
	# Manually ravel the data. This allows asymmetric replicate numbers.
	all_x_gens, all_y_cells = [], []
	for datum in data:
		for irep, rep in enumerate(datum):
			for igen, cell in enumerate(rep):
				all_x_gens.append(igen)
				all_y_cells.append(cell)
	all_x_gens = np.asfarray(all_x_gens)
	all_y_cells = np.asfarray(all_y_cells)
	all_Ndata = len(all_y_cells)
	orig_N0 = df['cells']['avg'][icnd][0]
	orig_model = Cyton15Model(hts, orig_N0, mgen, DT, nreps, lognorm)

	pos = mp.current_process()._identity[0]-1  # For progress bar
	tqdm_trange1 = tqdm.trange(ITER_BOOTS, leave=False, position=2*pos+1)
	for b in tqdm_trange1:
		sample_df = subsample(conv_df, sl)  # randomly sample replicates with replacement (select "sl" random samples per row independently)
		if sample_df.iloc[0].isnull().all(axis=None):  # check if all samples are None for first time point
			while sample_df.iloc[0].isnull().all(axis=None):  # resample until at least one of data point is not None
				sample_df = subsample(conv_df, sl)

		# Manually ravel the data. This allows asymmetric replicate numbers.
		x_gens, y_cells = [], []
		init, _hts, _nreps = True, [], []
		for idx, row in sample_df.iterrows():
			irep = 0
			for cgen in row:
				if cgen is not None:
					for igen, cell in enumerate(cgen):
						x_gens.append(igen)
						y_cells.append(cell)
					_hts.append(idx)
					irep += 1
			# check if the row is empty
			if not all(v is None for v in row):
				_nreps.append(irep)
				if init:
					init = False
					avgN0 = np.array(row.dropna().values.tolist()).mean(axis=0).sum()
		_hts = np.unique(_hts)
		x_gens = np.asfarray(x_gens)
		y_cells = np.asfarray(y_cells)

		params = lmf.Parameters()
		# LMFIT add parameter properties with tuples: (NAME, VALUE, VARY, MIN, MAX, EXPR, BRUTE_STEP)
		for par in pars:
			params.add(par, value=pars[par], min=bounds['lb'][par], max=bounds['ub'][par], vary=vary[par])
		paramExcl = [p for p in params if not params[p].vary]  # List of parameters excluded from fitting (i.e. vary=False)

		model = Cyton15Model(_hts, avgN0, mgen, DT, _nreps, lognorm)

		candidates = {'algo': [], 'result': [], 'residual': []}  # store fitted parameter and its residual
		tqdm_trange2 = tqdm.trange(ITER_SEARCH, leave=False, position=2*pos+2)
		for s in tqdm_trange2:
			# Random initial values
			for par in params:
				if par in paramExcl: pass  # Ignore excluded parameters
				else:
					par_min, par_max = params[par].min, params[par].max  # determine its min and max range
					params[par].set(value=rng.uniform(low=par_min, high=par_max))

			try:  # Some set of initial values is completely non-sensical, resulted in NaN errors
				mini_lm = lmf.Minimizer(residual, params, fcn_args=(x_gens, y_cells, model), **LM_FIT_KWS)
				res_lm = mini_lm.minimize(method='leastsq', max_nfev=MAX_NFEV)  # Levenberg-Marquardt algorithm
				# res_lm = mini_lm.minimize(method='least_squares', max_nfev=MAX_NFEV)  # Trust Region Reflective method

				algo = 'LM'  # record algorithm name
				result = res_lm
				resid = res_lm.chisqr

				tqdm_trange2.set_description(f"[SEARCH] > {key} > {''.join(condition.split()) + f' {sl} Reps'}")
				tqdm_trange2.set_postfix({'RSS': f"{resid:.5e}"})
				tqdm_trange2.refresh()

				candidates['algo'].append(algo)
				candidates['result'].append(result)
				candidates['residual'].append(resid)
			except ValueError as ve:
				tqdm_trange2.update()

		fit_results = pd.DataFrame(candidates)
		fit_results.sort_values('residual', ascending=True, inplace=True)  # sort based on residual

		# Extract best-fit parameters
		best_result = fit_results.iloc[0]['result']
		best_fit = best_result.params.valuesdict()
		for var in best_fit:
			boots[var].append(best_fit[var])
		boots['N0'].append(avgN0)
		boots['algo'].append(algo)

		mse = best_result.chisqr/best_result.ndata
		rmse = np.sqrt(mse)
		boots['mse-in'].append(mse)
		boots['rmse-in'].append(rmse)

		rss_out = np.sum(residual(best_result.params, all_x_gens, all_y_cells, orig_model)**2)
		mse_out = rss_out/all_Ndata
		rmse_out = np.sqrt(mse_out)
		boots['mse-out'].append(mse_out)
		boots['rmse-out'].append(rmse_out)
	boots_all = pd.DataFrame(boots)
	boots = pd.DataFrame(boots).drop(['mse-in', 'mse-out', 'rmse-in', 'rmse-out'], axis=1)

	## calculate average of parameters
	means = boots.mean()
	mUns, sUns = means['mUns'], means['sUns']
	mDiv0, sDiv0 = means['mDiv0'], means['sDiv0']
	mDD, sDD = means['mDD'], means['sDD']
	mDie, sDie = means['mDie'], means['sDie']
	m, p = means['m'], means['p']
	N0 = means['N0']

	mean_params = lmf.Parameters()
	mean_params.add_many(('mUns', mUns), ('sUns', sUns),
						 ('mDiv0', mDiv0), ('sDiv0', sDiv0),
						 ('mDD', mDD), ('sDD', sDD),
						 ('mDie', mDie), ('sDie', sDie),
						 ('m', m), ('p', p))

	### PLOT RESULTS
	t0, tf = 0, max(hts)+5
	times = np.linspace(t0, tf, num=int(tf/DT)+1)
	gens = np.array([i for i in range(mgen+1)])

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
	for bsample in boots.drop('algo', axis=1).iterrows():
		b_mUns, b_sUns, b_mDiv0, b_sDiv0, b_mDD, b_sDD, b_mDie, b_sDie, b_m, b_p, b_N0 = bsample[1].values
		b_params = params.copy()
		b_params['mUns'].set(value=b_mUns); b_params['sUns'].set(value=b_sUns)
		b_params['mDiv0'].set(value=b_mDiv0); b_params['sDiv0'].set(value=b_sDiv0)
		b_params['mDD'].set(value=b_mDD); b_params['sDD'].set(value=b_sDD)
		b_params['mDie'].set(value=b_mDie); b_params['sDie'].set(value=b_sDie)
		b_params['m'].set(value=b_m); b_params['p'].set(value=b_p)

		# Calculate PDF and CDF curves for each set of parameter
		if lognorm:
			b_unst_pdf, b_unst_cdf = lognorm_pdf(times, b_mUns, b_sUns), lognorm_cdf(times, b_mUns, b_sUns)
			b_tdiv0_pdf, b_tdiv0_cdf = lognorm_pdf(times, b_mDiv0, b_sDiv0), lognorm_cdf(times, b_mDiv0, b_sDiv0)
			b_tdd_pdf, b_tdd_cdf = lognorm_pdf(times, b_mDD, b_sDD), lognorm_cdf(times, b_mDD, b_sDD)
			b_tdie_pdf, b_tdie_cdf = lognorm_pdf(times, b_mDie, b_sDie), lognorm_cdf(times, b_mDie, b_sDie)
		else:
			b_unst_pdf, b_unst_cdf = norm_pdf(times, b_mUns, b_sUns), norm_cdf(times, b_mUns, b_sUns)
			b_tdiv0_pdf, b_tdiv0_cdf = norm_pdf(times, b_mDiv0, b_sDiv0), norm_cdf(times, b_mDiv0, b_sDiv0)
			b_tdd_pdf, b_tdd_cdf = norm_pdf(times, b_mDD, b_sDD), norm_cdf(times, b_mDD, b_sDD)
			b_tdie_pdf, b_tdie_cdf = norm_pdf(times, b_mDie, b_sDie), norm_cdf(times, b_mDie, b_sDie)

		unst_pdf_curves.append(b_unst_pdf); unst_cdf_curves.append(b_unst_cdf)
		tdiv0_pdf_curves.append(b_tdiv0_pdf); tdiv0_cdf_curves.append(b_tdiv0_cdf)
		tdd_pdf_curves.append(b_tdd_pdf); tdd_cdf_curves.append(b_tdd_cdf)
		tdie_pdf_curves.append(b_tdie_pdf); tdie_cdf_curves.append(b_tdie_cdf)

		# Calculate model prediction for each set of parameter
		b_model = Cyton15Model(hts, b_N0, mgen, DT, nreps, lognorm)
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

	# 95% confidence interval on each parameter values
	err_mUns, err_sUns = conf_iterval(boots['mUns'], RGS), conf_iterval(boots['sUns'], RGS)
	err_mDiv0, err_sDiv0 = conf_iterval(boots['mDiv0'], RGS), conf_iterval(boots['sDiv0'], RGS)
	err_mDD, err_sDD = conf_iterval(boots['mDD'], RGS), conf_iterval(boots['sDD'], RGS)
	err_mDie, err_sDie = conf_iterval(boots['mDie'], RGS), conf_iterval(boots['sDie'], RGS)
	err_m, err_p = conf_iterval(boots['m'], RGS), conf_iterval(boots['p'], RGS)

	tmp_err_N0 = conf_iterval(tmp_N0, RGS)  # AGAIN, NOT A REAL PARAMETER

	save_best_fit = pd.DataFrame(
		data={"mean": [mUns, sUns, mDiv0, sDiv0, mDD, sDD, mDie, sDie, m, p, N0],
			"low95": [mUns-err_mUns[0], sUns-err_sUns[0], mDiv0-err_mDiv0[0], sDiv0-err_sDiv0[0], mDD-err_mDD[0], sDD-err_sDD[0], mDie-err_mDie[0], sDie-err_sDie[0], m-err_m[0], p-err_p[0], N0-tmp_err_N0[0]],
			"high95": [err_mUns[1]-mUns, err_sUns[1]-sUns, err_mDiv0[1]-mDiv0, err_sDiv0[1]-sDiv0, err_mDD[1]-mDD, err_sDD[1]-sDD, err_mDie[1]-mDie, err_sDie[1]-sDie, err_m[1]-m, err_p[1]-p, tmp_err_N0[1]-N0],
			"vary": np.append([params[p].vary for p in params], "False")}, 
		index=["mUns", "sUns", "mDiv0", "sDiv0", "mDD", "sDD", "mDie", "sDie", "m", "p", "N0"]) 
	
	if lognorm:
		excel_path = f"./out/_lognormal/indiv/{key}_result.xlsx"
	else:
		excel_path = f"./out/_normal/indiv/{key}_result.xlsx"
	if os.path.isfile(excel_path): writer = pd.ExcelWriter(excel_path, engine='openpyxl', mode='a')
	else: writer = pd.ExcelWriter(excel_path, engine='openpyxl', mode='w')
	save_best_fit.to_excel(writer, sheet_name=f"pars_{sl}reps")
	boots_all.to_excel(writer, sheet_name=f"boot_{sl}reps")
	writer.save()
	writer.close()

	## Get extrapolation: 1. for given time range t \in [t0, tf]; 2. at harvested time points
	model = Cyton15Model(hts, N0, mgen, DT, nreps, lognorm)
	extrapolate = model.extrapolate(times, mean_params)  # get extrapolation for all "times" (discretised) and at harvested timepoints
	ext_total_live_cells = extrapolate['ext']['total_live_cells']
	ext_cells_per_gen = extrapolate['ext']['cells_gen']
	hts_total_live_cells = extrapolate['hts']['total_live_cells']
	hts_cells_per_gen = extrapolate['hts']['cells_gen']

	# Calculate PDF and CDF
	if lognorm:
		unst_pdf, unst_cdf = lognorm_pdf(times, mUns, sUns), lognorm_cdf(times, mUns, sUns)
		tdiv0_pdf, tdiv0_cdf = lognorm_pdf(times, mDiv0, sDiv0), lognorm_cdf(times, mDiv0, sDiv0)
		tdd_pdf, tdd_cdf = lognorm_pdf(times, mDD, sDD), lognorm_cdf(times, mDD, sDD)
		tdie_pdf, tdie_cdf = lognorm_pdf(times, mDie, sDie), lognorm_cdf(times, mDie, sDie)
	else:
		unst_pdf, unst_cdf = norm_pdf(times, mUns, sUns), norm_cdf(times, mUns, sUns)
		tdiv0_pdf, tdiv0_cdf = norm_pdf(times, mDiv0, sDiv0), norm_cdf(times, mDiv0, sDiv0)
		tdd_pdf, tdd_cdf = norm_pdf(times, mDD, sDD), norm_cdf(times, mDD, sDD)
		tdie_pdf, tdie_cdf = norm_pdf(times, mDie, sDie), norm_cdf(times, mDie, sDie)

	### FIG 1: SUMMARY PLOT
	fig1, ax1 = plt.subplots(nrows=2, ncols=2, sharex=True)
	fig1.suptitle(f"[{key}][{condition}][{sl} Reps] Cyton parameters, Total cohort and cell numbers")
	
	## PROBABILITY DISTRIBUTION FUNCTION
	ax1[0,0].set_title(f"$t_{{div}}={m:.2f}h \pm_{{{m-err_m[0]:.2f}}}^{{{err_m[1]-m:.2f}}}$")
	ax1[0,0].set_ylabel("Density")
	if lognorm:
		label_Tuns = f"$T_{{uns}} \sim \mathcal{{LN}}({mUns:.2f} \pm_{{{mUns-err_mUns[0]:.2f}}}^{{{err_mUns[1]-mUns:.2f}}}, {sUns:.3f} \pm_{{{sUns-err_sUns[0]:.3f}}}^{{{err_sUns[1]-sUns:.3f}}})$"
		label_Tdiv0 = f"$T_{{div}}^0 \sim \mathcal{{LN}}({mDiv0:.2f}\pm_{{{mDiv0-err_mDiv0[0]:.2f}}}^{{{err_mDiv0[1]-mDiv0:.2f}}}, {sDiv0:.3f} \pm_{{{sDiv0-err_sDiv0[0]:.3f}}}^{{{err_sDiv0[1]-sDiv0:.3f}}})$"
		label_Tdd = f"$T_{{dd}} \sim \mathcal{{LN}}({mDD:.2f}\pm_{{{mDD-err_mDD[0]:.2f}}}^{{{err_mDD[1]-mDD:.2f}}}, {sDD:.3f}\pm_{{{sDD-err_sDD[0]:.3f}}}^{{{err_sDD[1]-sDD:.3f}}})$"
		label_Tdie = f"$T_{{die}} \sim \mathcal{{LN}}({mDie:.2f}\pm_{{{mDie-err_mDie[0]:.2f}}}^{{{err_mDie[1]-mDie:.2f}}}, {sDie:.3f}\pm_{{{sDie-err_sDie[0]:.3f}}}^{{{err_sDie[1]-sDie:.3f}}})$"
	else:
		label_Tuns = f"$T_{{uns}} \sim \mathcal{{N}}({mUns:.2f} \pm_{{{mUns-err_mUns[0]:.2f}}}^{{{err_mUns[1]-mUns:.2f}}}, {sUns:.3f} \pm_{{{sUns-err_sUns[0]:.3f}}}^{{{err_sUns[1]-sUns:.3f}}})$"
		label_Tdiv0 = f"$T_{{div}}^0 \sim \mathcal{{N}}({mDiv0:.2f}\pm_{{{mDiv0-err_mDiv0[0]:.2f}}}^{{{err_mDiv0[1]-mDiv0:.2f}}}, {sDiv0:.3f} \pm_{{{sDiv0-err_sDiv0[0]:.3f}}}^{{{err_sDiv0[1]-sDiv0:.3f}}})$"
		label_Tdd = f"$T_{{dd}} \sim \mathcal{{N}}({mDD:.2f}\pm_{{{mDD-err_mDD[0]:.2f}}}^{{{err_mDD[1]-mDD:.2f}}}, {sDD:.3f}\pm_{{{sDD-err_sDD[0]:.3f}}}^{{{err_sDD[1]-sDD:.3f}}})$"
		label_Tdie = f"$T_{{die}} \sim \mathcal{{N}}({mDie:.2f}\pm_{{{mDie-err_mDie[0]:.2f}}}^{{{err_mDie[1]-mDie:.2f}}}, {sDie:.3f}\pm_{{{sDie-err_sDie[0]:.3f}}}^{{{err_sDie[1]-sDie:.3f}}})$"
	# ax1[0,0].plot(times, -unst_pdf, color='orange', ls='--', label=label_Tuns)
	# ax1[0,0].fill_between(times, -conf['unst_pdf'][0], -conf['unst_pdf'][1], fc='orange', ec=None, alpha=0.5)
	ax1[0,0].plot(times, tdiv0_pdf, color='blue', ls='-', label=label_Tdiv0)
	ax1[0,0].fill_between(times, conf['tdiv0_pdf'][0], conf['tdiv0_pdf'][1], fc='blue', ec=None, alpha=0.5)
	ax1[0,0].plot(times, tdd_pdf, color='green', ls='-', label=label_Tdd)
	ax1[0,0].fill_between(times, conf['tdd_pdf'][0], conf['tdd_pdf'][1], fc='green', ec=None, alpha=0.5)
	ax1[0,0].plot(times, -tdie_pdf, color='red', ls='-', label=label_Tdie)
	ax1[0,0].fill_between(times, -conf['tdie_pdf'][0], -conf['tdie_pdf'][1], fc='red', ec=None, alpha=0.5)
	ax1[0,0].set_yticklabels(np.round(np.abs(ax1[0,0].get_yticks()), 5))  # remove negative y-tick labels
	ax1[0,0].legend(fontsize=9, frameon=True)

	## CUMULATIVE DISTRIBUTION FUNCTION
	ax1[1,0].set_title(f"$p = {p:.4f} \pm_{{{p-err_p[0]:.4f}}}^{{{err_p[1]-p:.4f}}}$")
	ax1[1,0].set_ylabel("CDF")
	ax1[1,0].set_xlabel("Time (hour)")
	# ax1[1,0].plot(times, unst_cdf, color='orange', ls='--')
	# ax1[1,0].fill_between(times, conf['unst_cdf'][0], conf['unst_cdf'][1], fc='orange', ec=None, alpha=0.5)
	ax1[1,0].plot(times, tdiv0_cdf, color='blue', ls='-')
	ax1[1,0].fill_between(times, conf['tdiv0_cdf'][0], conf['tdiv0_cdf'][1], fc='blue', ec=None, alpha=0.5)
	ax1[1,0].plot(times, tdd_cdf, color='green', ls='-')
	ax1[1,0].fill_between(times, conf['tdd_cdf'][0], conf['tdd_cdf'][1], fc='green', ec=None, alpha=0.5)
	ax1[1,0].plot(times, tdie_cdf, color='red', ls='-')
	ax1[1,0].fill_between(times, conf['tdie_cdf'][0], conf['tdie_cdf'][1], fc='red', ec=None, alpha=0.5)
	ax1[1,0].set_ylim(bottom=0, top=1)

	## TOTAL COHORT NUMBER
	ax1[0,1].set_title("Total cohort number")
	ax1[0,1].set_ylabel("Cohort number")

	tps, total_cohorts = [], []
	for itpt, ht in enumerate(hts):
		for irep in range(nreps[itpt]):
			tps.append(ht)
			total_cohorts.append(np.sum(df['cohorts_gens']['rep'][icnd][itpt][irep]))
	ax1[0,1].plot(tps, total_cohorts, 'r.', label='data')

	ext_total_cohorts = np.sum(np.transpose(ext_cells_per_gen) * np.power(2.,-gens), axis=1)
	ax1[0,1].plot(times, ext_total_cohorts, 'k-', label='model')
	ax1[0,1].fill_between(times, conf['ext_total_cohorts'][0], conf['ext_total_cohorts'][1], fc='k', ec=None, alpha=0.3)
	ax1[0,1].set_ylim(bottom=0)
	ax1[0,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	ax1[0,1].yaxis.major.formatter._useMathText = True
	ax1[0,1].legend(fontsize=9, frameon=True)

	## TOTAL CELL NUMBERS 
	ax1[1,1].set_title(f"Total cell number: $N_0 = {np.round(N0)}$")
	ax1[1,1].set_ylabel("Cell number")
	ax1[1,1].set_xlabel("Time (hour)")

	tps, total_cells = [], []
	for itpt, ht in enumerate(hts):
		for irep in range(nreps[itpt]):
			tps.append(ht)
			total_cells.append(df['cells']['rep'][icnd][itpt][irep])
	ax1[1,1].plot(tps, total_cells, 'r.')

	ax1[1,1].plot(times, ext_total_live_cells, 'k-', lw=1)
	ax1[1,1].fill_between(times, conf['ext_total_live_cells'][0], conf['ext_total_live_cells'][1], fc='k', ec=None, alpha=0.3)
	cp = sns.hls_palette(mgen+1, l=0.4, s=0.5)
	for igen in range(mgen+1):
		ax1[1,1].errorbar(hts, np.transpose(df['cgens']['avg'][icnd])[igen], yerr=np.transpose(df['cgens']['sem'][icnd])[igen], c=cp[igen], fmt='.', ms=5, label=f"Gen {igen}")
		ax1[1,1].plot(times, ext_cells_per_gen[igen], c=cp[igen])
		ax1[1,1].fill_between(times, conf['ext_cells_per_gen'][0][igen], conf['ext_cells_per_gen'][1][igen], fc=cp[igen], ec=None, alpha=0.5)
	ax1[1,1].set_ylim(bottom=0)
	ax1[1,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	ax1[1,1].yaxis.major.formatter._useMathText = True
	# ax1[1,1].symlog()
	ax1[1,1].legend(fontsize=9, frameon=True)
	for ax in ax1.flat:
		ax.set_xlim(t0, max(times))
	fig1.tight_layout(rect=(0.0, 0.0, 1, 1))
	# fig1.subplots_adjust(hspace=0, wspace=0)

	### FIG 2: CELL NUMBERS PER GENERATION AT HARVESTED TIME POINTS
	if len(hts) <= 6: nrows, ncols = 2, 3
	elif 6 < len(hts) <= 9: nrows, ncols = 3, 3
	else: nrows, ncols = 4, 3

	fig2 = plt.figure()
	# fig2.suptitle(f"[{condition}][{sl} Reps] Cell numbers per generation at harvested time")
	fig2.text(0.5, 0.04, "Generations", ha='center', va='center')
	fig2.text(0.02, 0.5, "Cell number", ha='center', va='center', rotation=90)
	axes = []  # store axis
	for itpt, ht in enumerate(hts):
		ax2 = plt.subplot(nrows, ncols, itpt+1)
		ax2.set_axisbelow(True)
		ax2.plot(gens, hts_cells_per_gen[itpt], 'o-', c='k', ms=5, label='model')
		ax2.fill_between(gens, conf['hts_cells_per_gen'][0][itpt], conf['hts_cells_per_gen'][1][itpt], fc='k', ec=None, alpha=0.3)
		for irep in range(nreps[itpt]):
			ax2.plot(gens, df['cgens']['rep'][icnd][itpt][irep], 'r.', label='data')
		ax2.set_xticks(gens)
		ax2.annotate(f"{ht}h", xy=(0.75, 0.85), xycoords='axes fraction')
		ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
		ax2.yaxis.major.formatter._useMathText = True
		if itpt not in [len(hts)-3, len(hts)-2, len(hts)-1]:
			# ax2.get_xaxis().set_ticks([])
			ax2.set_xticklabels([])
		if itpt not in [0, 3, 6, 9, 12]:
			# ax2.get_yaxis().set_ticks([])
			ax2.set_yticklabels([])
		ax2.spines['right'].set_visible(True)
		ax2.spines['top'].set_visible(True)
		ax2.grid(True, ls='--')
		axes.append(ax2)
	max_ylim = 0
	for ax in axes:
		_, ymax = ax.get_ylim()
		if max_ylim < ymax:
			max_ylim = ymax
	for ax in axes:
		ax.set_ylim(top=max_ylim)
	handles, labels = ax.get_legend_handles_labels()
	fig2.legend(handles=[handles[1], handles[0]], labels=[labels[1], labels[0]], ncol=2, bbox_to_anchor=(1,1))
	fig2.tight_layout(rect=(0.02, 0.03, 1, 1))
	fig2.subplots_adjust(wspace=0.05, hspace=0.15)


	### FIG 3: DISTRIBUTION OF BOOTSTRAP SAMPLES 
	alpha = (1. - RGS/100.)/2
	# quantiles = boots[['mUns', 'sUns', 'mDiv0', 'sDiv0', 'mDD', 'sDD', 'mDie', 'sDie', 'm', 'p']].quantile([alpha, alpha + RGS/100.], numeric_only=True, interpolation='nearest')
	# titles = ["$T_{uns}$", "$T_{uns}$", "$T_{div}^0$", "$T_{div}^0$", "$T_{dd}$", "$T_{dd}$", "$T_{die}$", "$T_{die}$", "$t_{div}$", "$p$"]
	# if lognorm:
	# 	xlabs = ["median ($m$)", "shape ($s$)", "median ($m$)", "shape ($s$)", "median ($m$)", "shape ($s$)", "median ($m$)", "shape ($s$)", "Subsequent Division Time (hour)", "Prob."]
	# else:
	# 	xlabs = ["mean ($\mu$)", "std ($\sigma$)", "mean ($\mu$)", "std ($\sigma$)", "mean ($\mu$)", "std ($\sigma$)", "mean ($\mu$)", "std ($\sigma$)", "Subsequent Division Time (hour)", "Prob."]
	# colors = ['orange', 'orange', 'blue', 'blue', 'green', 'green', 'red', 'red', 'navy', 'k']
	# fig3, ax3 = plt.subplots(nrows=5, ncols=2, figsize=(9, 8))
	# fig3.suptitle(f"[{sl} Reps] Bootstrap marginal distribution")
	# ax3 = ax3.flat
	# for i, obj in enumerate(boots.drop(['N0', 'algo'], axis=1)):
	# 	best = best_fit[obj]
	# 	b_sample = boots[obj].to_numpy()
	# 	l_quant, h_quant = quantiles.iloc[0][obj], quantiles.iloc[1][obj]

	# 	ax3[i].set_title(titles[i])
	# 	ax3[i].axvline(best, ls='-', c='k', label=f"best-fit={best:.2f}")
	# 	ax3[i].axvline(l_quant, ls=':', c='red', label=f"lo={l_quant:.2f}")
	# 	ax3[i].axvline(h_quant, ls=':', c='red', label=f"hi={h_quant:.2f}")
	# 	sns.distplot(b_sample, kde=False, hist_kws=dict(ec='k', lw=1), color=colors[i], ax=ax3[i])
	# 	ax3[i].set_xlabel(xlabs[i])
	# 	ax3[i].legend(fontsize=9, loc='upper right')
	# fig3.tight_layout(rect=(0.01, 0, 1, 1))
	# fig3.subplots_adjust(wspace=0.1, hspace=0.83)

	quantiles = boots[['mDiv0', 'sDiv0', 'mDD', 'sDD', 'mDie', 'sDie', 'm', 'p']].quantile([alpha, alpha + RGS/100.], numeric_only=True, interpolation='nearest')
	titles = ["$T_{div}^0$", "$T_{div}^0$", "$T_{dd}$", "$T_{dd}$", "$T_{die}$", "$T_{die}$", "$t_{div}$", "$p$"]
	if lognorm:
		xlabs = ["median ($m$)", "shape ($s$)", "median ($m$)", "shape ($s$)", "median ($m$)", "shape ($s$)", "Subsequent Division Time (hour)", "Prob."]
	else:
		xlabs = ["mean ($\mu$)", "std ($\sigma$)", "mean ($\mu$)", "std ($\sigma$)", "mean ($\mu$)", "std ($\sigma$)", "Subsequent Division Time (hour)", "Prob."]
	colors = ['blue', 'blue', 'green', 'green', 'red', 'red', 'navy', 'k']
	fig3, ax3 = plt.subplots(nrows=4, ncols=2, figsize=(9, 8))
	fig3.suptitle(f"[{sl} Reps] Bootstrap marginal distribution")
	ax3 = ax3.flat
	for i, obj in enumerate(boots.drop(['mUns', 'sUns', 'N0', 'algo'], axis=1)):
		mean = means[obj]
		b_sample = boots[obj].to_numpy()
		l_quant, h_quant = quantiles.iloc[0][obj], quantiles.iloc[1][obj]

		ax3[i].set_title(titles[i])
		ax3[i].axvline(mean, ls='-', c='k', label=f"$\mu$={mean:.2f}")
		ax3[i].axvline(l_quant, ls=':', c='red', label=f"lo={l_quant:.2f}")
		ax3[i].axvline(h_quant, ls=':', c='red', label=f"hi={h_quant:.2f}")
		sns.distplot(b_sample, kde=False, hist_kws=dict(ec='k', lw=1), color=colors[i], ax=ax3[i])
		ax3[i].set_xlabel(xlabs[i])
		ax3[i].legend(fontsize=9, loc='upper right')
	fig3.tight_layout(rect=(0.01, 0, 1, 1))
	fig3.subplots_adjust(wspace=0.1, hspace=0.83)


	### FIG 4: OTHER WAY TO PLOT BOOTSTRAP SAMPLES
	# fig4, ax4 = plt.subplots(nrows=2, ncols=3)
	# fig4.suptitle(f"[{condition}][{sl} Reps] Bootstrap distribution")
	# sns.distplot(boots['p'], hist_kws=dict(ec='k', lw=1), kde=False, norm_hist=True, color='k', ax=ax4[0,0])
	# ax4[0,0].axvline(best_fit['p'], ls='-', c='k', label=f"best-fit={best_fit['p']:.4f}")
	# ax4[0,0].axvline(quantiles.iloc[0]['p'], ls=':', c='red', label=f"lo={quantiles.iloc[0]['p']:.4f}")
	# ax4[0,0].axvline(quantiles.iloc[1]['p'], ls=':', c='red', label=f"hi={quantiles.iloc[1]['p']:.4f}")
	# ax4[0,0].set_title("Activation probability")
	# ax4[0,0].set_ylabel("Frequency")
	# ax4[0,0].set_xlabel("Probability")
	# ax4[0,0].legend(fontsize=9, frameon=True, loc='upper right')

	# sns.distplot(boots['m'], hist_kws=dict(ec='k', lw=1), kde=False, norm_hist=True, color='navy', ax=ax4[1,0])
	# ax4[1,0].axvline(best_fit['m'], ls='-', c='k', label=f"best-fit={best_fit['m']:.2f}")
	# ax4[1,0].axvline(quantiles.iloc[0]['m'], ls=':', c='red', label=f"lo={quantiles.iloc[0]['m']:.2f}")
	# ax4[1,0].axvline(quantiles.iloc[1]['m'], ls=':', c='red', label=f"hi={quantiles.iloc[1]['m']:.2f}")
	# ax4[1,0].set_title("Subsequent division time")
	# ax4[1,0].set_ylabel("Frequency")
	# ax4[1,0].set_xlabel("Time (hour)")
	# ax4[1,0].legend(fontsize=9, frameon=True, loc='upper right')

	# if lognorm:
	# 	notat1, notat2 = "m", "s"
	# 	ylab_Tuns = "shape, $s$ ($T_{uns}$)"; xlab_Tuns = "median, $m$ ($T_{uns}$)"
	# 	ylab_Tdiv0 = "shape, $s$ ($T_{div}^0$)"; xlab_Tdiv0 = "median, $m$ ($T_{div}^0$)"
	# 	ylab_Tdd = "shape, $s$ ($T_{dd}$)"; xlab_Tdd = "median, $m$ ($T_{dd}$)"
	# 	ylab_Tdie = "shape, $s$ ($T_{die}$)"; xlab_Tdie = "median, $m$ ($T_{die}$)"
	# else: 
	# 	notat1, notat2 = "\mu", "\sigma"
	# 	ylab_Tuns = "std, $\sigma$ ($T_{uns}$)"; xlab_Tuns = "mean, $\mu$ ($T_{uns}$)"
	# 	ylab_Tdiv0 = "std, $\sigma$ ($T_{div}^0$)"; xlab_Tdiv0 = "mean, $\mu$ ($T_{div}^0$)"
	# 	ylab_Tdd = "std, $\sigma$ ($T_{dd}$)"; xlab_Tdd = "mean, $\mu$ ($T_{dd}$)"
	# 	ylab_Tdie = "std, $\sigma$ ($T_{die}$)"; xlab_Tdie = "mean, $\mu$ ($T_{die}$)"
	# sns.scatterplot(x=boots['mUns'], y=boots['sUns'], color='orange', ec=None, linewidth=1, alpha=0.5, ax=ax4[0,1])
	# ax4[0,1].errorbar(x=mUns, y=sUns, xerr=[[mUns-err_mUns[0]], [err_mUns[1]-mUns]], yerr=[[sUns-err_sUns[0]], [err_sUns[1]-sUns]], fmt='.', color='k', alpha=0.7, label=f"$m = {mUns:.2f}\pm_{{{mUns-err_mUns[0]:.2f}}}^{{{err_mUns[1]-mUns:.2f}}}$\n" + f"$s = {sUns:.3f}\pm_{{{sUns-err_sUns[0]:.3f}}}^{{{err_sUns[1]-sUns:.3f}}}$")
	# ax4[0,1].set_title("Time to death (unstimulated)")
	# ax4[0,1].set_ylabel(ylab_Tuns)
	# ax4[0,1].set_xlabel(xlab_Tuns)
	# ax4[0,1].spines['right'].set_visible(True)
	# ax4[0,1].spines['top'].set_visible(True)
	# ax4[0,1].legend(fontsize=9, frameon=True, loc='upper right')

	# sns.scatterplot(x=boots['mDiv0'], y=boots['sDiv0'], color='blue', ec=None, linewidth=1, alpha=0.5, ax=ax4[0,2])
	# ax4[0,2].errorbar(x=mDiv0, y=sDiv0, xerr=[[mDiv0-err_mDiv0[0]], [err_mDiv0[1]-mDiv0]], yerr=[[sDiv0-err_sDiv0[0]], [err_sDiv0[1]-sDiv0]], fmt='.', color='k', alpha=0.7, label=f"$m = {mDiv0:.2f}\pm_{{{mDiv0-err_mDiv0[0]:.2f}}}^{{{err_mDiv0[1]-mDiv0:.2f}}}$\n" + f"$s = {sDiv0:.3f}\pm_{{{sDiv0-err_sDiv0[0]:.3f}}}^{{{err_sDiv0[1]-sDiv0:.3f}}}$")
	# ax4[0,2].set_title("Time to first division")
	# ax4[0,2].set_ylabel(ylab_Tdiv0)
	# ax4[0,2].set_xlabel(xlab_Tdiv0)
	# ax4[0,2].spines['right'].set_visible(True)
	# ax4[0,2].spines['top'].set_visible(True)
	# ax4[0,2].legend(fontsize=9, frameon=True, loc='upper right')

	# sns.scatterplot(x=boots['mDD'], y=boots['sDD'], color='green', ec=None, linewidth=1, alpha=0.5, ax=ax4[1,1])
	# ax4[1,1].errorbar(x=mDD, y=sDD, xerr=[[mDD-err_mDD[0]], [err_mDD[1]-mDD]], yerr=[[sDD-err_sDD[0]], [err_sDD[1]-sDD]], fmt='.', color='k', alpha=0.7, label=f"${notat1} = {mDD:.2f}\pm_{{{mDD-err_mDD[0]:.2f}}}^{{{err_mDD[1]-mDD:.2f}}}$\n" + f"${notat2} = {sDD:.3f}\pm_{{{sDD-err_sDD[0]:.3f}}}^{{{err_sDD[1]-sDD:.3f}}}$")
	# ax4[1,1].set_title("Time to division destiny")
	# ax4[1,1].set_ylabel(ylab_Tdd)
	# ax4[1,1].set_xlabel(xlab_Tdd)
	# ax4[1,1].spines['right'].set_visible(True)
	# ax4[1,1].spines['top'].set_visible(True)
	# ax4[1,1].legend(fontsize=9, frameon=True, loc='upper right')

	# sns.scatterplot(x=boots['mDie'], y=boots['sDie'], color='red', ec=None, linewidth=1, alpha=0.5, ax=ax4[1,2])
	# ax4[1,2].errorbar(x=mDie, y=sDie, xerr=[[mDie-err_mDie[0]], [err_mDie[1]-mDie]], yerr=[[sDie-err_sDie[0]], [err_sDie[1]-sDie]], fmt='.', color='k', alpha=0.7, label=f"${notat1} = {mDie:.2f}\pm_{{{mDie-err_mDie[0]:.2f}}}^{{{err_mDie[1]-mDie:.2f}}}$\n" + f"${notat2} = {sDie:.3f}\pm_{{{sDie-err_sDie[0]:.3f}}}^{{{err_sDie[1]-sDie:.3f}}}$")
	# ax4[1,2].set_title("Time to death")
	# ax4[1,2].set_ylabel(ylab_Tdie)
	# ax4[1,2].set_xlabel(xlab_Tdie)
	# ax4[1,2].spines['right'].set_visible(True)
	# ax4[1,2].spines['top'].set_visible(True)
	# ax4[1,2].legend(fontsize=9, frameon=True, loc='upper right')
	# fig4.tight_layout(rect=(0.01, 0, 1, 1))
	# # fig4.subplots_adjust(wspace=0.05, hspace=0.05)

	# ### FIG 5: CORRELATION PLOT (PAIRPLOT)
	# grid = sns.pairplot(boots.drop(np.append(paramExcl, ['N0', 'algo']), axis=1), markers="o", palette=sns.color_palette(['#003f5c']), height=1, corner=True, diag_kind='hist', diag_kws=dict(ec='k', lw=1, alpha=0.5), plot_kws=dict(s=22, ec=None, linewidth=1, alpha=0.5), grid_kws=dict(diag_sharey=False))
	# grid.fig.suptitle(f"[{condition}][{sl} Reps] Parameter correlation")
	# # grid.fig.set_size_inches(14, 10)
	# grid.fig.set_size_inches(rc['figure.figsize'][0], rc['figure.figsize'][1])
	# grid.fig.tight_layout()

	fig4, ax4 = plt.subplots(nrows=2, ncols=2)
	fig4.suptitle(f"[{condition}][{sl} Reps] Bootstrap distribution")
	sns.distplot(boots['m'], hist_kws=dict(ec='k', lw=1), kde=False, norm_hist=True, color='navy', ax=ax4[0,0])
	ax4[0,0].axvline(best_fit['m'], ls='-', c='k', label=f"best-fit={best_fit['m']:.2f}")
	ax4[0,0].axvline(quantiles.iloc[0]['m'], ls=':', c='red', label=f"lo={quantiles.iloc[0]['m']:.2f}")
	ax4[0,0].axvline(quantiles.iloc[1]['m'], ls=':', c='red', label=f"hi={quantiles.iloc[1]['m']:.2f}")
	ax4[0,0].set_title("Subsequent division time")
	ax4[0,0].set_ylabel("Frequency")
	ax4[0,0].set_xlabel("Time (hour)")
	ax4[0,0].legend(fontsize=9, frameon=True, loc='upper right')

	if lognorm:
		notat1, notat2 = "m", "s"
		ylab_Tdiv0 = "shape, $s$ ($T_{div}^0$)"; xlab_Tdiv0 = "median, $m$ ($T_{div}^0$)"
		ylab_Tdd = "shape, $s$ ($T_{dd}$)"; xlab_Tdd = "median, $m$ ($T_{dd}$)"
		ylab_Tdie = "shape, $s$ ($T_{die}$)"; xlab_Tdie = "median, $m$ ($T_{die}$)"
	else: 
		notat1, notat2 = "\mu", "\sigma"
		ylab_Tdiv0 = "std, $\sigma$ ($T_{div}^0$)"; xlab_Tdiv0 = "mean, $\mu$ ($T_{div}^0$)"
		ylab_Tdd = "std, $\sigma$ ($T_{dd}$)"; xlab_Tdd = "mean, $\mu$ ($T_{dd}$)"
		ylab_Tdie = "std, $\sigma$ ($T_{die}$)"; xlab_Tdie = "mean, $\mu$ ($T_{die}$)"

	sns.scatterplot(x=boots['mDiv0'], y=boots['sDiv0'], color='blue', ec=None, linewidth=1, alpha=0.5, ax=ax4[0,1])
	ax4[0,1].errorbar(x=mDiv0, y=sDiv0, xerr=[[mDiv0-err_mDiv0[0]], [err_mDiv0[1]-mDiv0]], yerr=[[sDiv0-err_sDiv0[0]], [err_sDiv0[1]-sDiv0]], fmt='.', color='k', alpha=0.7, label=f"$m = {mDiv0:.2f}\pm_{{{mDiv0-err_mDiv0[0]:.2f}}}^{{{err_mDiv0[1]-mDiv0:.2f}}}$\n" + f"$s = {sDiv0:.3f}\pm_{{{sDiv0-err_sDiv0[0]:.3f}}}^{{{err_sDiv0[1]-sDiv0:.3f}}}$")
	ax4[0,1].set_title("Time to first division")
	ax4[0,1].set_ylabel(ylab_Tdiv0)
	ax4[0,1].set_xlabel(xlab_Tdiv0)
	ax4[0,1].spines['right'].set_visible(True)
	ax4[0,1].spines['top'].set_visible(True)
	ax4[0,1].legend(fontsize=9, frameon=True, loc='upper right')

	sns.scatterplot(x=boots['mDD'], y=boots['sDD'], color='green', ec=None, linewidth=1, alpha=0.5, ax=ax4[1,0])
	ax4[1,0].errorbar(x=mDD, y=sDD, xerr=[[mDD-err_mDD[0]], [err_mDD[1]-mDD]], yerr=[[sDD-err_sDD[0]], [err_sDD[1]-sDD]], fmt='.', color='k', alpha=0.7, label=f"${notat1} = {mDD:.2f}\pm_{{{mDD-err_mDD[0]:.2f}}}^{{{err_mDD[1]-mDD:.2f}}}$\n" + f"${notat2} = {sDD:.3f}\pm_{{{sDD-err_sDD[0]:.3f}}}^{{{err_sDD[1]-sDD:.3f}}}$")
	ax4[1,0].set_title("Time to division destiny")
	ax4[1,0].set_ylabel(ylab_Tdd)
	ax4[1,0].set_xlabel(xlab_Tdd)
	ax4[1,0].spines['right'].set_visible(True)
	ax4[1,0].spines['top'].set_visible(True)
	ax4[1,0].legend(fontsize=9, frameon=True, loc='upper right')

	sns.scatterplot(x=boots['mDie'], y=boots['sDie'], color='red', ec=None, linewidth=1, alpha=0.5, ax=ax4[1,1])
	ax4[1,1].errorbar(x=mDie, y=sDie, xerr=[[mDie-err_mDie[0]], [err_mDie[1]-mDie]], yerr=[[sDie-err_sDie[0]], [err_sDie[1]-sDie]], fmt='.', color='k', alpha=0.7, label=f"${notat1} = {mDie:.2f}\pm_{{{mDie-err_mDie[0]:.2f}}}^{{{err_mDie[1]-mDie:.2f}}}$\n" + f"${notat2} = {sDie:.3f}\pm_{{{sDie-err_sDie[0]:.3f}}}^{{{err_sDie[1]-sDie:.3f}}}$")
	ax4[1,1].set_title("Time to death")
	ax4[1,1].set_ylabel(ylab_Tdie)
	ax4[1,1].set_xlabel(xlab_Tdie)
	ax4[1,1].spines['right'].set_visible(True)
	ax4[1,1].spines['top'].set_visible(True)
	ax4[1,1].legend(fontsize=9, frameon=True, loc='upper right')
	fig4.tight_layout(rect=(0.01, 0, 1, 1))
	# fig4.subplots_adjust(wspace=0.05, hspace=0.05)

	### FIG 5: CORRELATION PLOT (PAIRPLOT)
	grid = sns.pairplot(boots.drop(np.append(paramExcl, ['N0', 'algo']), axis=1), markers="o", palette=sns.color_palette(['#003f5c']), height=1, corner=True, diag_kind='hist', diag_kws=dict(ec='k', lw=1, alpha=0.5), plot_kws=dict(s=22, ec=None, linewidth=1, alpha=0.5), grid_kws=dict(diag_sharey=False))
	grid.fig.suptitle(f"[{condition}][{sl} Reps] Parameter correlation")
	# grid.fig.set_size_inches(14, 10)
	grid.fig.set_size_inches(rc['figure.figsize'][0], rc['figure.figsize'][1])
	grid.fig.tight_layout()

	if lognorm:
		with PdfPages(f"./out/_lognormal/indiv/{key}_{sl}reps.pdf") as pdf:
			pdf.savefig(fig1)
			pdf.savefig(fig2)
			pdf.savefig(fig3)
			pdf.savefig(fig4)
			pdf.savefig(grid.fig)
	else:
		with PdfPages(f"./out/_normal/indiv/{key}_{sl}reps.pdf") as pdf:
			pdf.savefig(fig1)
			pdf.savefig(fig2)
			pdf.savefig(fig3)
			pdf.savefig(fig4)
			pdf.savefig(grid.fig)

if __name__ == "__main__":
	start = time.time()
	print('> No. of BOOTSTRAP ITERATIONS: {0}'.format(ITER_BOOTS))
	print('> No. of SEARCH ITERATIONS for CYTON FITTING: {0}'.format(ITER_SEARCH))

	## IMPORT DATA
	PATH_TO_DATA = './data'
	DATA = ["SH1.119.xlsx"]  # 9 replicate B cell data
	KEYS = [os.path.splitext(os.path.basename(data_key))[0] for data_key in DATA]
	df = parse_data(PATH_TO_DATA, DATA)

	## ANALYSE EFFECT OF REPLICATE NUMBER
	inputs = []
	data_slice = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # number of replicates
	for key in KEYS:
		reader = df[key]['reader']
		for sl in data_slice:
			inputs.append((key, df[key], reader, sl, LOGNORM))
	tqdm.tqdm.set_lock(mp.RLock())  # for managing output contention
	p = mp.Pool(initializer=tqdm.tqdm.set_lock, initargs=(tqdm.tqdm.get_lock(),))
	with tqdm.tqdm(total=len(inputs), desc="Total", position=0) as pbar:
		for i, _ in enumerate(p.imap_unordered(fit_reps, inputs)):
			pbar.update()
	p.close()
	p.join()

	## REPORT ELAPSED TIME
	end = time.time()
	hours, rem = divmod(end-start, 3600)
	minutes, seconds = divmod(rem, 60)
	now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
	print(f"> DONE FITTING ! {now}")
	print("> Elapsed Time = {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))