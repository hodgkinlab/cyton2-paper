"""
Last edit: 16-May-2021

Application of the Cyton2 model: T cell signal integration
Data received from Dr. Julia M. Marchingo. (Fig.3 DOI:10.1126/science.1260044)
OT-1/Bcl2l11-/- CD8+ T cells stimulated with [N4(medium), aCD27, aCD28, IL-12, aCD27+IL-12, aCD27+aCD28, aCD28+IL-12, aCD27+aCD28+IL-12]
Joint fitting script for N4, aCD27, aCD28 with a shared subsequent division time
[Output] Data files (in excel format) for Fig6 -> Run "Fig6-Plot.py" for generating the plots
"""
import sys, os, time, datetime, copy
import tqdm
import numpy as np
import pandas as pd
import matplotlib as mpl; mpl.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import multiprocessing as mp
import lmfit as lmf
from src.parse import parse_data
from src.utils import conf_iterval, norm_pdf, norm_cdf, lognorm_pdf, lognorm_cdf
from src.model import Cyton2Model   # Full Cyton model. (Options: choose all variable to be either logN or N)
rng = np.random.RandomState(seed=54755083)

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
ITER_SEARCH = 200   # [Cyton Model] Number of initial search
MAX_NFEV = None     # [LMFIT] Maximum number of function evaluation
LM_FIT_KWS = {      # [LMFIT/SciPy] Key-word arguements pass to LMFIT minimizer for Levenberg-Marquardt algorithm
	# 'ftol': 1E-10,  # Relative error desired in the sum of squares. DEFAULT: 1.49012E-8
	# 'xtol': 1E-10,  # Relative error desired in the approximate solution. DEFAULT: 1.49012E-8
	# 'gtol': 0.0,    # Orthogonality desired between the function vector and the columns of the Jacobian. DEFAULT: 0.0
	'epsfcn': 1E-4  # A variable used in determining a suitable step length for the forward-difference approximation of the Jacobian (for Dfun=None). Normally the actual step length will be sqrt(epsfcn)*x If epsfcn is less than the machine precision, it is assumed that the relative errors are of the order of the machine precision. Default value is around 2E-16. As it turns out, the optimisation routine starts by making a very small move (functinal evaluation) and calculating the finite-difference Jacobian matrix to determine the direction to move. The default value is too small to detect sensitivity of 'm' parameter.
}

LOGNORM = False	 	# All variables ~ LN(m,s); Otherwisee, all variables ~ N(mu,sig)
RGS = 95            # [BOOTSTRAP] alpha for confidence interval
ITER_BOOTS = 1000   # [BOOTSTRAP] Bootstrap samples

### CYTON MODEL: OBJECTIVE FUNCTION
def residual(pars, x, data=None, model=None):
	ndata = len(data)
	resid = []
	vals = pars.valuesdict()
	m = vals['m']
	for i in range(ndata):
		mUns, sUns = vals[f'mUns_{i}'], vals[f'sUns_{i}']
		mDiv0, sDiv0 = vals[f'mDiv0_{i}'], vals[f'sDiv0_{i}']
		mDD, sDD = vals[f'mDD_{i}'], vals[f'sDD_{i}']
		mDie, sDie = vals[f'mDie_{i}'], vals[f'sDie_{i}']
		p = vals[f'p_{i}']

		pred = model[i].evaluate(mUns, sUns, mDiv0, sDiv0, mDD, sDD, mDie, sDie, m, p)
		resid = np.append(resid, np.array(data[i]) - pred)
	return resid.flatten()

def bootstrap(key, df, targets, hts, nreps, params, paramExcl, lognorm):
	# define set collectors
	boots = {'algo': []}
	for par in params:
		boots[par] = []
	for icnd, _ in enumerate(targets):
		boots[f'N0_{icnd}'] = []
	
	pred_aCD27_aCD28 = {
		'mUns': [], 'sUns': [],
		'mDiv0': [], 'sDiv0': [],
		'mDD': [], 'sDD': [],
		'mDie': [], 'sDie': [],
		'm': [], 'p': []
	}

	# for b in range(ITER_BOOTS):
	pos = mp.current_process()._identity[0]-1  # For progress bar
	tqdm_trange1 = tqdm.trange(ITER_BOOTS, leave=False, position=2*pos+1)
	for b in tqdm_trange1:
		tqdm_trange1.set_description(f"[BOOTSTRAP]")
		tqdm_trange1.update()
		pars = copy.copy(params)

		x_boot, y_boot, N0 = [], [], [] # generations & raveled n(g,t)
		for icnd, _ in enumerate(targets):
			cells = df['cgens']['rep'][icnd]  # n(g,t): number of cells per gen at t
			first = 1
			tmp_xboot, tmp_yboot = [], []
			for data in cells:
				tmp_N0 = 0
				for _ in range(len(data)):
					rand_idx = rng.randint(0, len(data))
					for igen, cell_number_rep in enumerate(data[rand_idx]):
						tmp_xboot.append(igen)
						tmp_yboot.append(cell_number_rep)
						tmp_N0 += cell_number_rep
				if first:
					N0.append(tmp_N0 / len(data))
					first = 0
			x_boot.append(tmp_xboot)
			y_boot.append(tmp_yboot)
		# convert from python lists to numpy arrays
		x_boot = np.array(x_boot)  # generation numbers
		y_boot = np.array(y_boot)  # number of cells per generation

		models = []
		for icnd, _ in enumerate(targets):
			model = Cyton2Model(hts[icnd], N0[icnd], int(max(x_boot[icnd])), DT, nreps[icnd], lognorm)  # define cyton model object
			models.append(model)

		err_count = 0
		candidates = {'algo': [], 'result': [], 'residual': []}  # store fitted parameter and its residual
		tqdm_trange2 = tqdm.trange(ITER_SEARCH, leave=False, position=2*pos+2)
		for s in tqdm_trange2:
			# Random initial values
			for par in pars:
				if par in paramExcl: pass  # Ignore excluded parameters
				else:
					par_min, par_max = pars[par].min, pars[par].max  # determine its min and max range
					pars[par].set(value=rng.uniform(low=par_min, high=par_max))

			try:  # Some set of initial values is completely non-sensical, resulted in NaN errors
				mini_lm = lmf.Minimizer(residual, pars, fcn_args=(x_boot, y_boot, models), **LM_FIT_KWS)
				res_lm = mini_lm.minimize(method='leastsq', max_nfev=MAX_NFEV)  # Levenberg-Marquardt algorithm
				# res_lm = mini_lm.minimize(method='least_squares', max_nfev=MAX_NFEV)  # Trust Region Reflective method

				algo = 'LM'
				result = res_lm
				resid = res_lm.chisqr

				tqdm_trange2.set_description(f" > {key} > {targets}")
				tqdm_trange2.set_postfix({'RSS': f"{resid:.5e}"})
				tqdm_trange2.refresh()

				candidates['algo'].append(algo)
				candidates['result'].append(result)
				candidates['residual'].append(resid)
			except ValueError as ve:
				err_count += 1
				tqdm_trange2.update()

		fit_results = pd.DataFrame(candidates)
		fit_results.sort_values('residual', ascending=True, inplace=True)  # sort based on residual

		boot_best_fit = fit_results.iloc[0]['result'].params.valuesdict()

		for var in boot_best_fit:
			boots[var].append(boot_best_fit[var])

		for icnd, _ in enumerate(targets):
			boots[f'N0_{icnd}'].append(N0[icnd])
		boots['algo'].append(fit_results.iloc[0]['algo'])

		## Save predicted aCD27 + aCD28
		shared_m = boot_best_fit['m']

		N4_mDiv0, N4_varDiv0 = boot_best_fit['mDiv0_0'], boot_best_fit['sDiv0_0']**2
		N4_mDD, N4_varDD = boot_best_fit['mDD_0'], boot_best_fit['sDD_0']**2
		N4_mDie, N4_varDie = boot_best_fit['mDie_0'], boot_best_fit['sDie_0']**2

		aCD27_mDiv0, aCD27_varDiv0 = boot_best_fit['mDiv0_1'], boot_best_fit['sDiv0_1']**2
		aCD27_mDD, aCD27_varDD = boot_best_fit['mDD_1'], boot_best_fit['sDD_1']**2
		aCD27_mDie, aCD27_varDie = boot_best_fit['mDie_1'], boot_best_fit['sDie_1']**2

		aCD28_mDiv0, aCD28_varDiv0 = boot_best_fit['mDiv0_2'], boot_best_fit['sDiv0_2']**2
		aCD28_mDD, aCD28_varDD = boot_best_fit['mDD_2'], boot_best_fit['sDD_2']**2
		aCD28_mDie, aCD28_varDie = boot_best_fit['mDie_2'], boot_best_fit['sDie_2']**2

		delta = {
			'aCD27': {
				'mDiv0': aCD27_mDiv0 - N4_mDiv0, 'vDiv0': aCD27_varDiv0 - N4_varDiv0,
				'mDD': aCD27_mDD - N4_mDD, 'vDD': aCD27_varDD - N4_varDD,
				'mDie': aCD27_mDie - N4_mDie, 'vDie': aCD27_varDie - N4_varDie
			},
			'aCD28': {
				'mDiv0': aCD28_mDiv0 - N4_mDiv0, 'vDiv0': aCD28_varDiv0 - N4_varDiv0,
				'mDD': aCD28_mDD - N4_mDD, 'vDD': aCD28_varDD - N4_varDD,
				'mDie': aCD28_mDie - N4_mDie, 'vDie': aCD28_varDie - N4_varDie
			}
		}
		pred_aCD27_aCD28['mUns'].append(boot_best_fit['mUns_0'])
		pred_aCD27_aCD28['sUns'].append(boot_best_fit['sUns_0'])
		pred_aCD27_aCD28['mDiv0'].append(N4_mDiv0 + delta['aCD27']['mDiv0'] + delta['aCD28']['mDiv0'])
		pred_aCD27_aCD28['sDiv0'].append(np.sqrt(N4_varDiv0 + delta['aCD27']['vDiv0'] + delta['aCD28']['vDiv0']))
		pred_aCD27_aCD28['mDD'].append(N4_mDD + delta['aCD27']['mDD'] + delta['aCD28']['mDD'])
		pred_aCD27_aCD28['sDD'].append(np.sqrt(N4_varDD + delta['aCD27']['vDD'] + delta['aCD28']['vDD']))
		pred_aCD27_aCD28['mDie'].append(N4_mDie + delta['aCD27']['mDie'] + delta['aCD28']['mDD'])
		pred_aCD27_aCD28['sDie'].append(np.sqrt(N4_varDie + delta['aCD27']['vDie'] + delta['aCD28']['vDie']))
		pred_aCD27_aCD28['m'].append(shared_m)
		pred_aCD27_aCD28['p'].append(1)
	boots = pd.DataFrame(boots)
	preds = pd.DataFrame(pred_aCD27_aCD28)
	return boots, preds


def joint_fit(inputs):
	key, df, lognorm = inputs

	reader = df['reader']
	conditions = reader.condition_names
	hts = reader.harvested_times
	mgen = reader.generation_per_condition

	target_conditions = conditions[:3]

	# Loop through N4 (signal 1), aCD27 (signal 2b), aCD28 (signal 2a)
	nreps = []
	x_gens, y_cells = [], []
	for icnd, _ in enumerate(target_conditions):
		### PREPARE DATA
		data = df['cgens']['rep'][icnd]  # n(g,t): number of cells in generation g at time t
		# Manually ravel the data. This allows asymmetric replicate numbers.
		tmp_nreps = []
		tmp_xgens, tmp_ycells = [], []
		for datum in data:
			for irep, rep in enumerate(datum):
				for igen, cell in enumerate(rep):
					tmp_xgens.append(igen)
					tmp_ycells.append(cell)
			tmp_nreps.append(irep+1)
		# Stack the data
		nreps.append(tmp_nreps)
		x_gens.append(tmp_xgens)
		y_cells.append(tmp_ycells)
	x_gens = np.array(x_gens)
	y_cells = np.array(y_cells)

	### ANALYSIS: ESTIMATE DIVISION PARAMETERS BY FITTING CYTON MODEL
	pars = {  # Initial values
			'mUns': 1000, 'sUns': 1E-3,  # Unstimulated death time (NOT USED HERE)
			'mDiv0': 30, 'sDiv0': 0.2,   # Time to first division
			'mDD': 60, 'sDD': 0.3,    	 # Time to division destiny
			'mDie': 80, 'sDie': 0.4,	 # Time to death
			'm': 10, 'p': 1				 # Subseqeunt division time & Activation probability (ASSUME ALL CELLS ACTIVATED)
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

	params = lmf.Parameters()
	# LMFIT add parameter properties
	for icnd, _ in enumerate(target_conditions):
		for par in pars:
			if not par == 'm':
				name = f"{par}_{icnd}"
			else:
				name = par
			params.add(name, value=pars[par], min=bounds['lb'][par], max=bounds['ub'][par], vary=vary[par])
	paramExcl = [p for p in params if not params[p].vary]  # List of parameters excluded from fitting (i.e. vary=False)

	models = []
	for icnd, _ in enumerate(target_conditions):
		N0 = df['cells']['avg'][icnd][0]  # Initial cell number = # cells measured at the first time point
		model = Cyton2Model(hts[icnd], N0, mgen[icnd], DT, nreps[icnd], lognorm)
		models.append(model)

	err_count = 0
	candidates = {'algo': [], 'result': [], 'residual': []}  # store fitted parameter and its residual
	pos = mp.current_process()._identity[0]-1  # For progress bar
	tqdm_trange = tqdm.trange(ITER_SEARCH, leave=False, position=2*pos+1)
	for s in tqdm_trange:
		# Random initial values
		for par in params:
			if par in paramExcl: pass  # Ignore excluded parameters
			else:
				par_min, par_max = params[par].min, params[par].max  # determine its min and max range
				params[par].set(value=rng.uniform(low=par_min, high=par_max))

		try:  # Some set of initial values is completely non-sensical, resulted in NaN errors
			mini_lm = lmf.Minimizer(residual, params, fcn_args=(x_gens, y_cells, models), **LM_FIT_KWS)
			res_lm = mini_lm.minimize(method='leastsq', max_nfev=MAX_NFEV)  # Levenberg-Marquardt algorithm
			# res_lm = mini_lm.minimize(method='least_squares', max_nfev=MAX_NFEV)  # Trust Region Reflective method

			algo = 'LM'
			result = res_lm
			resid = res_lm.chisqr

			tqdm_trange.set_description(f"[SEARCH] > {key} > {target_conditions}")
			tqdm_trange.set_postfix({'RSS': f"{resid:.5e}"})
			tqdm_trange.refresh()

			candidates['algo'].append(algo)
			candidates['result'].append(result)
			candidates['residual'].append(resid)
		except ValueError as ve:
			err_count += 1

	fit_results = pd.DataFrame(candidates)
	fit_results.sort_values('residual', ascending=True, inplace=True)  # sort based on residual
	best_fit = fit_results.iloc[0]['result'].params.valuesdict()  # Extract best-fit parameters

	### RUN BOOTSTRAP
	boots, preds = bootstrap(key, df, target_conditions, hts, nreps, params, paramExcl, lognorm)

	## Calculate prediction for aCD27 + aCD28
	shared_m = best_fit['m']

	N4_mDiv0, N4_varDiv0 = best_fit['mDiv0_0'], best_fit['sDiv0_0']**2
	N4_mDD, N4_varDD = best_fit['mDD_0'], best_fit['sDD_0']**2
	N4_mDie, N4_varDie = best_fit['mDie_0'], best_fit['sDie_0']**2

	aCD27_mDiv0, aCD27_varDiv0 = best_fit['mDiv0_1'], best_fit['sDiv0_1']**2
	aCD27_mDD, aCD27_varDD = best_fit['mDD_1'], best_fit['sDD_1']**2
	aCD27_mDie, aCD27_varDie = best_fit['mDie_1'], best_fit['sDie_1']**2

	aCD28_mDiv0, aCD28_varDiv0 = best_fit['mDiv0_2'], best_fit['sDiv0_2']**2
	aCD28_mDD, aCD28_varDD = best_fit['mDD_2'], best_fit['sDD_2']**2
	aCD28_mDie, aCD28_varDie = best_fit['mDie_2'], best_fit['sDie_2']**2

	delta = {
		'aCD27': {
			'mDiv0': aCD27_mDiv0 - N4_mDiv0, 'vDiv0': aCD27_varDiv0 - N4_varDiv0,
			'mDD': aCD27_mDD - N4_mDD, 'vDD': aCD27_varDD - N4_varDD,
			'mDie': aCD27_mDie - N4_mDie, 'vDie': aCD27_varDie - N4_varDie
		},
		'aCD28': {
			'mDiv0': aCD28_mDiv0 - N4_mDiv0, 'vDiv0': aCD28_varDiv0 - N4_varDiv0,
			'mDD': aCD28_mDD - N4_mDD, 'vDD': aCD28_varDD - N4_varDD,
			'mDie': aCD28_mDie - N4_mDie, 'vDie': aCD28_varDie - N4_varDie
		}
	}
	aCD27_aCD28 = {
		'mUns': best_fit['mUns_0'], 'sUns': best_fit['sUns_0'],
		'mDiv0': N4_mDiv0 + delta['aCD27']['mDiv0'] + delta['aCD28']['mDiv0'], 'sDiv0': np.sqrt(N4_varDiv0 + delta['aCD27']['vDiv0'] + delta['aCD28']['vDiv0']),
		'mDD': N4_mDD + delta['aCD27']['mDD'] + delta['aCD28']['mDD'], 'sDD': np.sqrt(N4_varDD + delta['aCD27']['vDD'] + delta['aCD28']['vDD']),
		'mDie': N4_mDie + delta['aCD27']['mDie'] + delta['aCD28']['mDD'], 'sDie': np.sqrt(N4_varDie + delta['aCD27']['vDie'] + delta['aCD28']['vDie']),
		'm': shared_m, 'p': 1
	}

	# 95% confidence interval on aCD27 + aCD28 prediction
	err_mUns, err_sUns = conf_iterval(preds['mUns'], RGS), conf_iterval(preds['sUns'], RGS)
	err_mDiv0, err_sDiv0 = conf_iterval(preds['mDiv0'], RGS), conf_iterval(preds['sDiv0'], RGS)
	err_mDD, err_sDD = conf_iterval(preds['mDD'], RGS), conf_iterval(preds['sDD'], RGS)
	err_mDie, err_sDie = conf_iterval(preds['mDie'], RGS), conf_iterval(preds['sDie'], RGS)
	err_m, err_p = conf_iterval(preds['m'], RGS), conf_iterval(preds['p'], RGS)
	save_pred = pd.DataFrame(
		data={
			"best-fit": [aCD27_aCD28['mUns'], aCD27_aCD28['sUns'], aCD27_aCD28['mDiv0'], aCD27_aCD28['sDiv0'], aCD27_aCD28['mDD'], aCD27_aCD28['sDD'], aCD27_aCD28['mDie'], aCD27_aCD28['sDie'], aCD27_aCD28['m'], aCD27_aCD28['p']],
			"low95": [aCD27_aCD28['mUns']-err_mUns[0], aCD27_aCD28['sUns']-err_sUns[0], aCD27_aCD28['mDiv0']-err_mDiv0[0], aCD27_aCD28['sDiv0']-err_sDiv0[0], aCD27_aCD28['mDD']-err_mDD[0], aCD27_aCD28['sDD']-err_sDD[0], aCD27_aCD28['mDie']-err_mDie[0], aCD27_aCD28['sDie']-err_sDie[0], aCD27_aCD28['m']-err_m[0], aCD27_aCD28['p']-err_p[0]],
			"high95": [err_mUns[1]-aCD27_aCD28['mUns'], err_sUns[1]-aCD27_aCD28['sUns'], err_mDiv0[1]-aCD27_aCD28['mDiv0'], err_sDiv0[1]-aCD27_aCD28['sDiv0'], err_mDD[1]-aCD27_aCD28['mDD'], err_sDD[1]-aCD27_aCD28['sDD'], err_mDie[1]-aCD27_aCD28['mDie'], err_sDie[1]-aCD27_aCD28['sDie'], err_m[1]-aCD27_aCD28['m'], err_p[1]-aCD27_aCD28['p']],
			"vary": [v for p, v in vary.items()]}, 
		index=["mUns", "sUns", "mDiv0", "sDiv0", "mDD", "sDD", "mDie", "sDie", "m", "p"]
	)
	if lognorm:
		excel_path = f"./out/_lognormal/joint/{key}_result.xlsx"
	else:
		excel_path = f"./out/_normal/joint/{key}_result.xlsx"
	if os.path.isfile(excel_path): 
		writer = pd.ExcelWriter(excel_path, engine='openpyxl', mode='a')
	else: 
		writer = pd.ExcelWriter(excel_path, engine='openpyxl', mode='w')
	save_pred.to_excel(writer, sheet_name="pars_pred_aCD27_aCD28")
	pred_boots = preds[['mUns', 'sUns', 'mDiv0', 'sDiv0', 'mDD', 'sDD', 'mDie', 'sDie', 'm', 'p']]
	pred_boots.to_excel(writer, sheet_name="boot_pred_aCD27_aCD28")
	writer.save()
	writer.close()

	for icnd, cond in enumerate(target_conditions):
		mUns, sUns = best_fit[f'mUns_{icnd}'], best_fit[f'sUns_{icnd}']
		mDiv0, sDiv0 = best_fit[f'mDiv0_{icnd}'], best_fit[f'sDiv0_{icnd}']
		mDD, sDD = best_fit[f'mDD_{icnd}'], best_fit[f'sDD_{icnd}']
		mDie, sDie = best_fit[f'mDie_{icnd}'], best_fit[f'sDie_{icnd}']
		m, p = best_fit['m'], best_fit[f'p_{icnd}']
		N0 = df['cells']['avg'][icnd][0]

		best_fit_params = lmf.Parameters()
		best_fit_params.add('mUns', value=mUns); best_fit_params.add('sUns', value=sUns)
		best_fit_params.add('mDiv0', value=mDiv0); best_fit_params.add('sDiv0', value=sDiv0)
		best_fit_params.add('mDD', value=mDD); best_fit_params.add('sDD', value=sDD)
		best_fit_params.add('mDie', value=mDie); best_fit_params.add('sDie', value=sDie)
		best_fit_params.add('m', value=m); best_fit_params.add('p', value=p)

		### PLOT RESULTS
		t0, tf = 0, max(hts[icnd])+5
		times = np.linspace(t0, tf, num=int(tf/DT)+1)
		gens = np.array([i for i in range(mgen[icnd]+1)])

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
		cols = boots.columns[boots.columns.to_series().str.contains(f'_{icnd}')].append(pd.Index(['m']))
		for bsample in boots[cols].iterrows():
			b_mUns, b_sUns = bsample[1][f'mUns_{icnd}'], bsample[1][f'sUns_{icnd}']
			b_mDiv0, b_sDiv0 = bsample[1][f'mDiv0_{icnd}'], bsample[1][f'sDiv0_{icnd}']
			b_mDD, b_sDD = bsample[1][f'mDD_{icnd}'], bsample[1][f'sDD_{icnd}']
			b_mDie, b_sDie = bsample[1][f'mDie_{icnd}'], bsample[1][f'sDie_{icnd}']
			b_m, b_p = bsample[1]['m'], bsample[1][f'p_{icnd}']
			b_N0 = bsample[1][f'N0_{icnd}']

			b_params = best_fit_params.copy()
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
			b_model = Cyton2Model(hts[icnd], b_N0, mgen[icnd], DT, nreps[icnd], lognorm)
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
		err_mUns, err_sUns = conf_iterval(boots[f'mUns_{icnd}'], RGS), conf_iterval(boots[f'sUns_{icnd}'], RGS)
		err_mDiv0, err_sDiv0 = conf_iterval(boots[f'mDiv0_{icnd}'], RGS), conf_iterval(boots[f'sDiv0_{icnd}'], RGS)
		err_mDD, err_sDD = conf_iterval(boots[f'mDD_{icnd}'], RGS), conf_iterval(boots[f'sDD_{icnd}'], RGS)
		err_mDie, err_sDie = conf_iterval(boots[f'mDie_{icnd}'], RGS), conf_iterval(boots[f'sDie_{icnd}'], RGS)
		err_m, err_p = conf_iterval(boots['m'], RGS), conf_iterval(boots[f'p_{icnd}'], RGS)

		tmp_err_N0 = conf_iterval(tmp_N0, RGS)  # AGAIN, NOT A REAL PARAMETER

		save_best_fit = pd.DataFrame(
			data={
				"best-fit": [mUns, sUns, mDiv0, sDiv0, mDD, sDD, mDie, sDie, m, p, N0],
				"low95": [mUns-err_mUns[0], sUns-err_sUns[0], mDiv0-err_mDiv0[0], sDiv0-err_sDiv0[0], mDD-err_mDD[0], sDD-err_sDD[0], mDie-err_mDie[0], sDie-err_sDie[0], m-err_m[0], p-err_p[0], N0-tmp_err_N0[0]],
				"high95": [err_mUns[1]-mUns, err_sUns[1]-sUns, err_mDiv0[1]-mDiv0, err_sDiv0[1]-sDiv0, err_mDD[1]-mDD, err_sDD[1]-sDD, err_mDie[1]-mDie, err_sDie[1]-sDie, err_m[1]-m, err_p[1]-p, tmp_err_N0[1]-N0],
				"vary": np.append([v for p, v in vary.items()], "False")}, 
			index=["mUns", "sUns", "mDiv0", "sDiv0", "mDD", "sDD", "mDie", "sDie", "m", "p", "N0"])
		if os.path.isfile(excel_path): 
			writer = pd.ExcelWriter(excel_path, engine='openpyxl', mode='a')
		else: 
			writer = pd.ExcelWriter(excel_path, engine='openpyxl', mode='w')
		save_best_fit.to_excel(writer, sheet_name=f"pars_{cond}")

		sboots = boots[[f'mUns_{icnd}', f'sUns_{icnd}', f'mDiv0_{icnd}', f'sDiv0_{icnd}', f'mDD_{icnd}', f'sDD_{icnd}', f'mDie_{icnd}', f'sDie_{icnd}', 'm', f'p_{icnd}', f'N0_{icnd}', 'algo']]
		sboots.to_excel(writer, sheet_name=f"boot_{cond}")
		writer.save()
		writer.close()
		# print(f"\n-----> [{key}] FIT REPORT")
		# print(lmf.fit_report(fit_results.iloc[0]['result']))
		# print(f"-----> [{key}][{cond}] SUMMARY")
		# print(save_best_fit, end='\n\n')

		# Get extrapolation: 1. for given time range t \in [t0, tf]; 2. at harvested time points
		extrapolate = models[icnd].extrapolate(times, best_fit_params)  # get extrapolation for all "times" (discretised) and at harvested timepoints
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
		fig1.suptitle(f"[{key}][{cond}] Cyton parameters, Total cohort and cell numbers")
		
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
		for itpt, ht in enumerate(hts[icnd]):
			for irep in range(nreps[icnd][itpt]):
				tps.append(ht)
				total_cohorts.append(np.sum(df['cohorts_gens']['rep'][icnd][itpt][irep]))
		ext_total_cohorts = np.sum(np.transpose(ext_cells_per_gen) * np.power(2.,-gens), axis=1)
		ax1[0,1].plot(tps, total_cohorts, 'r.', label='data')
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
		for itpt, ht in enumerate(hts[icnd]):
			for irep in range(nreps[icnd][itpt]):
				tps.append(ht)
				total_cells.append(df['cells']['rep'][icnd][itpt][irep])
		ax1[1,1].plot(tps, total_cells, 'r.')
		ax1[1,1].plot(times, ext_total_live_cells, 'k-', lw=1)
		ax1[1,1].fill_between(times, conf['ext_total_live_cells'][0], conf['ext_total_live_cells'][1], fc='k', ec=None, alpha=0.3)
		cp = sns.hls_palette(mgen[icnd]+1, l=0.4, s=0.5)
		for igen in range(mgen[icnd]+1):
			ax1[1,1].errorbar(hts[icnd], np.transpose(df['cgens']['avg'][icnd])[igen], yerr=np.transpose(df['cgens']['sem'][icnd])[igen], c=cp[igen], fmt='.', ms=5, label=f"Gen {igen}")
			ax1[1,1].plot(times, ext_cells_per_gen[igen], c=cp[igen])
			ax1[1,1].fill_between(times, conf['ext_cells_per_gen'][0][igen], conf['ext_cells_per_gen'][1][igen], fc=cp[igen], ec=None, alpha=0.5)
		ax1[1,1].set_ylim(bottom=0)
		ax1[1,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
		ax1[1,1].yaxis.major.formatter._useMathText = True
		# ax1[1,1].symlog()
		ax1[1,1].legend(fontsize=9, frameon=True)
		for ax in ax1.flat:
			ax.set_xlim(t0, max(times))
		# fig1.subplots_adjust(hspace=0, wspace=0)
		fig1.tight_layout(rect=(0.0, 0.0, 1, 1))

		### FIG 2: CELL NUMBERS PER GENERATION AT HARVESTED TIME POINTS
		if len(hts[icnd]) <= 6: nrows, ncols = 2, 3
		elif 6 < len(hts[icnd]) <= 9: nrows, ncols = 3, 3
		else: nrows, ncols = 4, 3

		fig2 = plt.figure()
		# fig2.suptitle(f"[{cond}] Cell numbers per generation at harvested time")
		fig2.text(0.5, 0.04, "Generations", ha='center', va='center')
		fig2.text(0.02, 0.5, "Cell number", ha='center', va='center', rotation=90)
		axes = []  # store axis
		for itpt, ht in enumerate(hts[icnd]):
			ax2 = plt.subplot(nrows, ncols, itpt+1)
			ax2.set_axisbelow(True)
			ax2.plot(gens, hts_cells_per_gen[itpt], 'o-', c='k', ms=5, label='model')
			ax2.fill_between(gens, conf['hts_cells_per_gen'][0][itpt], conf['hts_cells_per_gen'][1][itpt], fc='k', ec=None, alpha=0.3)
			for irep in range(nreps[icnd][itpt]):
				ax2.plot(gens, df['cgens']['rep'][icnd][itpt][irep], 'r.', label='data')
			ax2.set_xticks(gens)
			ax2.annotate(f"{ht}h", xy=(0.75, 0.85), xycoords='axes fraction')
			ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
			ax2.yaxis.major.formatter._useMathText = True
			if itpt not in [len(hts[icnd])-3, len(hts[icnd])-2, len(hts[icnd])-1]:
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
		quantiles = boots[[f'mUns_{icnd}', f'sUns_{icnd}', f'mDiv0_{icnd}', f'sDiv0_{icnd}', f'mDD_{icnd}', f'sDD_{icnd}', f'mDie_{icnd}', f'sDie_{icnd}', 'm', f'p_{icnd}']].quantile([alpha, alpha + RGS/100.], numeric_only=True, interpolation='nearest')
		
		titles = ["$T_{uns}$", "$T_{uns}$", "$T_{div}^0$", "$T_{div}^0$", "$T_{dd}$", "$T_{dd}$", "$T_{die}$", "$T_{die}$", "$t_{div}$", "$p$"]
		if lognorm:
			xlabs = ["median ($m$)", "shape ($s$)", "median ($m$)", "shape ($s$)", "median ($m$)", "shape ($s$)", "median ($m$)", "shape ($s$)", "Subsequent Division Time (hour)", "Prob."]
		else:
			xlabs = ["mean ($\mu$)", "std ($\sigma$)", "mean ($\mu$)", "std ($\sigma$)", "mean ($\mu$)", "std ($\sigma$)", "mean ($\mu$)", "std ($\sigma$)", "Subsequent Division Time (hour)", "Prob."]
		colors = ['orange', 'orange', 'blue', 'blue', 'green', 'green', 'red', 'red', 'navy', 'k']
		fig3, ax3 = plt.subplots(nrows=5, ncols=2, figsize=(9, 8))
		fig3.suptitle(f"[{cond}] Bootstrap marginal distribution")
		ax3 = ax3.flat

		sboots = boots[[f'mUns_{icnd}', f'sUns_{icnd}', f'mDiv0_{icnd}', f'sDiv0_{icnd}', f'mDD_{icnd}', f'sDD_{icnd}', f'mDie_{icnd}', f'sDie_{icnd}', 'm', f'p_{icnd}']]
		for i, obj in enumerate(sboots):
			best = best_fit[obj]
			b_sample = boots[obj].to_numpy()
			l_quant, h_quant = quantiles.iloc[0][obj], quantiles.iloc[1][obj]

			ax3[i].set_title(titles[i])
			ax3[i].axvline(best, ls='-', c='k', label=f"best-fit={best:.2f}")
			ax3[i].axvline(l_quant, ls=':', c='red', label=f"lo={l_quant:.2f}")
			ax3[i].axvline(h_quant, ls=':', c='red', label=f"hi={h_quant:.2f}")
			sns.distplot(b_sample, kde=False, hist_kws=dict(ec='k', lw=1), color=colors[i], ax=ax3[i])
			ax3[i].set_xlabel(xlabs[i])
			ax3[i].legend(fontsize=9, loc='upper right')
		fig3.tight_layout(rect=(0.01, 0, 1, 1))
		fig3.subplots_adjust(wspace=0.1, hspace=0.83)


		### FIG 4: OTHER WAY TO PLOT BOOTSTRAP SAMPLES
		fig4, ax4 = plt.subplots(nrows=2, ncols=3)
		fig4.suptitle(f"[{cond}] Bootstrap distribution")
		sns.distplot(boots[f'p_{icnd}'], hist_kws=dict(ec='k', lw=1), kde=False, norm_hist=True, color='k', ax=ax4[0,0])
		ax4[0,0].axvline(p, ls='-', c='k', label=f"best-fit={p:.4f}")
		ax4[0,0].axvline(quantiles.iloc[0][f'p_{icnd}'], ls=':', c='red', label=f"lo={quantiles.iloc[0][f'p_{icnd}']:.4f}")
		ax4[0,0].axvline(quantiles.iloc[1][f'p_{icnd}'], ls=':', c='red', label=f"hi={quantiles.iloc[1][f'p_{icnd}']:.4f}")
		ax4[0,0].set_title("Activation probability")
		ax4[0,0].set_ylabel("Frequency")
		ax4[0,0].set_xlabel("Probability")
		ax4[0,0].legend(fontsize=9, frameon=True, loc='upper right')

		sns.distplot(boots['m'], hist_kws=dict(ec='k', lw=1), kde=False, norm_hist=True, color='navy', ax=ax4[1,0])
		ax4[1,0].axvline(m, ls='-', c='k', label=f"best-fit={m:.2f}")
		ax4[1,0].axvline(quantiles.iloc[0]['m'], ls=':', c='red', label=f"lo={quantiles.iloc[0]['m']:.2f}")
		ax4[1,0].axvline(quantiles.iloc[1]['m'], ls=':', c='red', label=f"hi={quantiles.iloc[1]['m']:.2f}")
		ax4[1,0].set_title("Subsequent division time")
		ax4[1,0].set_ylabel("Frequency")
		ax4[1,0].set_xlabel("Time (hour)")
		ax4[1,0].legend(fontsize=9, frameon=True, loc='upper right')

		if lognorm:
			notat1, notat2 = "m", "s"
			ylab_Tuns = "shape, $s$ ($T_{uns}$)"; xlab_Tuns = "median, $m$ ($T_{uns}$)"
			ylab_Tdiv0 = "shape, $s$ ($T_{div}^0$)"; xlab_Tdiv0 = "median, $m$ ($T_{div}^0$)"
			ylab_Tdd = "shape, $s$ ($T_{dd}$)"; xlab_Tdd = "median, $m$ ($T_{dd}$)"
			ylab_Tdie = "shape, $s$ ($T_{die}$)"; xlab_Tdie = "median, $m$ ($T_{die}$)"
		else: 
			notat1, notat2 = "\mu", "\sigma"
			ylab_Tuns = "std, $\sigma$ ($T_{uns}$)"; xlab_Tuns = "mean, $\mu$ ($T_{uns}$)"
			ylab_Tdiv0 = "std, $\sigma$ ($T_{div}^0$)"; xlab_Tdiv0 = "mean, $\mu$ ($T_{div}^0$)"
			ylab_Tdd = "std, $\sigma$ ($T_{dd}$)"; xlab_Tdd = "mean, $\mu$ ($T_{dd}$)"
			ylab_Tdie = "std, $\sigma$ ($T_{die}$)"; xlab_Tdie = "mean, $\mu$ ($T_{die}$)"
		sns.scatterplot(x=boots[f'mUns_{icnd}'], y=boots[f'sUns_{icnd}'], color='orange', ec=None, linewidth=1, alpha=0.5, ax=ax4[0,1])
		# ax4[0,1].scatter(x=best_fit['mUns'], y=best_fit['sUns'], color='k', alpha=0.7, label=f"best_fit = ({best_fit['mUns']:.2f}, {best_fit['sUns']:.3f})")
		ax4[0,1].errorbar(x=mUns, y=sUns, xerr=[[mUns-err_mUns[0]], [err_mUns[1]-mUns]], yerr=[[sUns-err_sUns[0]], [err_sUns[1]-sUns]], fmt='.', color='k', alpha=0.7, label=f"$m = {mUns:.2f}\pm_{{{mUns-err_mUns[0]:.2f}}}^{{{err_mUns[1]-mUns:.2f}}}$\n" + f"$s = {sUns:.3f}\pm_{{{sUns-err_sUns[0]:.3f}}}^{{{err_sUns[1]-sUns:.3f}}}$")
		ax4[0,1].set_title("Time to death (unstimulated)")
		ax4[0,1].set_ylabel(ylab_Tuns)
		ax4[0,1].set_xlabel(xlab_Tuns)
		ax4[0,1].spines['right'].set_visible(True)
		ax4[0,1].spines['top'].set_visible(True)
		ax4[0,1].legend(fontsize=9, frameon=True, loc='upper right')

		sns.scatterplot(x=boots[f'mDiv0_{icnd}'], y=boots[f'sDiv0_{icnd}'], color='blue', ec=None, linewidth=1, alpha=0.5, ax=ax4[0,2])
		# ax4[0,2].scatter(x=best_fit['mDiv0'], y=best_fit['sDiv0'], color='k', alpha=0.7, label=f"best_fit = ({best_fit['mDiv0']:.2f}, {best_fit['sDiv0']:.3f})")
		ax4[0,2].errorbar(x=mDiv0, y=sDiv0, xerr=[[mDiv0-err_mDiv0[0]], [err_mDiv0[1]-mDiv0]], yerr=[[sDiv0-err_sDiv0[0]], [err_sDiv0[1]-sDiv0]], fmt='.', color='k', alpha=0.7, label=f"$m = {mDiv0:.2f}\pm_{{{mDiv0-err_mDiv0[0]:.2f}}}^{{{err_mDiv0[1]-mDiv0:.2f}}}$\n" + f"$s = {sDiv0:.3f}\pm_{{{sDiv0-err_sDiv0[0]:.3f}}}^{{{err_sDiv0[1]-sDiv0:.3f}}}$")
		ax4[0,2].set_title("Time to first division")
		ax4[0,2].set_ylabel(ylab_Tdiv0)
		ax4[0,2].set_xlabel(xlab_Tdiv0)
		ax4[0,2].spines['right'].set_visible(True)
		ax4[0,2].spines['top'].set_visible(True)
		ax4[0,2].legend(fontsize=9, frameon=True, loc='upper right')

		sns.scatterplot(x=boots[f'mDD_{icnd}'], y=boots[f'sDD_{icnd}'], color='green', ec=None, linewidth=1, alpha=0.5, ax=ax4[1,1])
		# ax4[1,1].scatter(x=best_fit['mDD'], y=best_fit['sDD'], color='k', alpha=0.7, label=f"best_fit = ({best_fit['mDD']:.2f}, {best_fit['sDD']:.3f})")
		ax4[1,1].errorbar(x=mDD, y=sDD, xerr=[[mDD-err_mDD[0]], [err_mDD[1]-mDD]], yerr=[[sDD-err_sDD[0]], [err_sDD[1]-sDD]], fmt='.', color='k', alpha=0.7, label=f"${notat1} = {mDD:.2f}\pm_{{{mDD-err_mDD[0]:.2f}}}^{{{err_mDD[1]-mDD:.2f}}}$\n" + f"${notat2} = {sDD:.3f}\pm_{{{sDD-err_sDD[0]:.3f}}}^{{{err_sDD[1]-sDD:.3f}}}$")
		ax4[1,1].set_title("Time to division destiny")
		ax4[1,1].set_ylabel(ylab_Tdd)
		ax4[1,1].set_xlabel(xlab_Tdd)
		ax4[1,1].spines['right'].set_visible(True)
		ax4[1,1].spines['top'].set_visible(True)
		ax4[1,1].legend(fontsize=9, frameon=True, loc='upper right')

		sns.scatterplot(x=boots[f'mDie_{icnd}'], y=boots[f'sDie_{icnd}'], color='red', ec=None, linewidth=1, alpha=0.5, ax=ax4[1,2])
		# ax4[1,2].scatter(x=best_fit['mDie'], y=best_fit['sDie'], color='k', alpha=0.7, label=f"best_fit = ({best_fit['mDie']:.2f}, {best_fit['sDie']:.3f})")
		ax4[1,2].errorbar(x=mDie, y=sDie, xerr=[[mDie-err_mDie[0]], [err_mDie[1]-mDie]], yerr=[[sDie-err_sDie[0]], [err_sDie[1]-sDie]], fmt='.', color='k', alpha=0.7, label=f"${notat1} = {mDie:.2f}\pm_{{{mDie-err_mDie[0]:.2f}}}^{{{err_mDie[1]-mDie:.2f}}}$\n" + f"${notat2} = {sDie:.3f}\pm_{{{sDie-err_sDie[0]:.3f}}}^{{{err_sDie[1]-sDie:.3f}}}$")
		ax4[1,2].set_title("Time to death")
		ax4[1,2].set_ylabel(ylab_Tdie)
		ax4[1,2].set_xlabel(xlab_Tdie)
		ax4[1,2].spines['right'].set_visible(True)
		ax4[1,2].spines['top'].set_visible(True)
		ax4[1,2].legend(fontsize=9, frameon=True, loc='upper right')
		fig4.tight_layout(rect=(0.01, 0, 1, 1))
		# fig4.subplots_adjust(wspace=0.05, hspace=0.05)


		### FIG 5: CORRELATION PLOT (PAIRPLOT)
		grid = sns.pairplot(boots[[f'mDiv0_{icnd}', f'sDiv0_{icnd}', f'mDD_{icnd}', f'sDD_{icnd}', f'mDie_{icnd}', f'sDie_{icnd}', 'm']], markers="o", palette=sns.color_palette(['#003f5c']), height=1, corner=True, diag_kind='hist', diag_kws=dict(ec='k', lw=1, alpha=0.5), plot_kws=dict(s=22, ec=None, linewidth=1, alpha=0.5), grid_kws=dict(diag_sharey=False))
		grid.fig.suptitle(f"[{cond}] Parameter correlation")
		# grid.fig.set_size_inches(14, 10)
		grid.fig.set_size_inches(rc['figure.figsize'][0], rc['figure.figsize'][1])
		grid.fig.tight_layout()

		if lognorm:
			with PdfPages(f"./out/_lognormal/joint/{key}_{cond}.pdf") as pdf:
				pdf.savefig(fig1)
				pdf.savefig(fig2)
				pdf.savefig(fig3)
				pdf.savefig(fig4)
				pdf.savefig(grid.fig)
		else:
			with PdfPages(f"./out/_normal/joint/{key}_{cond}.pdf") as pdf:
				pdf.savefig(fig1)
				pdf.savefig(fig2)
				pdf.savefig(fig3)
				pdf.savefig(fig4)
				pdf.savefig(grid.fig)


if __name__ == "__main__":
	start = time.time()
	print('> No. of BOOTSTRAP ITERATIONS: {0}'.format(ITER_BOOTS))
	print('> No. of SEARCH ITERATIONS for CYTON FITTING: {0}'.format(ITER_SEARCH))

	PATH_TO_DATA = './data'
	DATA = ["EX127.xlsx", "EX130b.xlsx"]
	KEYS = [os.path.splitext(os.path.basename(data_key))[0] for data_key in DATA]
	
	df = parse_data(PATH_TO_DATA, DATA)

	inputs = []
	for key in KEYS:
		reader = df[key]['reader']
		inputs.append((key, df[key], LOGNORM))

	tqdm.tqdm.set_lock(mp.RLock())  # for managing output contention
	p = mp.Pool(initializer=tqdm.tqdm.set_lock, initargs=(tqdm.tqdm.get_lock(),))
	with tqdm.tqdm(total=len(inputs), desc="Data Files", position=0) as pbar:
		for i, _ in enumerate(p.imap_unordered(joint_fit, inputs)):
			pbar.update()
	p.close()
	p.join()

	end = time.time()
	hours, rem = divmod(end-start, 3600)
	minutes, seconds = divmod(rem, 60)
	now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
	print(f"> DONE FITTING ! {now}")
	print("> Elapsed Time = {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))