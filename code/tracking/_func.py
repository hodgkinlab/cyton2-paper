"""
Last edit: 16-November-2020

Functions to process the filming data.
Extra plot tools
"""
import os, copy
import numpy as np
import pandas as pd
import scipy.io as sio

def parse_data(paths):
	df_exp = {}
	keys = []
	df_template = {
		'clone': [], 'gen': [], 'relation': [], 'col_birth': [], 't_birth': [], 't_rOn': [], 't_rOff': [], 
		't_gOn': [], 't_gOff': [], 't_div': [], 't_death': [], 't_loss': [], 'lifetime': [], 'fate': [],
		'stim': [], 'well': []
	}
	for path in paths:
		# load .mat data
		event = sio.loadmat(os.path.join(path, 'events.mat'), squeeze_me=True)
		
		# sort data in data frame
		fams = event['family']
		fams_size = len(event['family'])
		vals = np.asarray(event['values'])
		names = event['cellNames']  # well information
		df = copy.deepcopy(df_template)

		df['col_birth'] = vals[:,0]
		df['t_birth'] = vals[:,1]
		df['t_rOn'] = vals[:,2]
		df['t_rOff'] = vals[:,3]
		df['t_gOn'] = vals[:,4]
		df['t_gOff'] = vals[:,5]
		df['t_div'] = vals[:,6]
		df['t_death'] = vals[:,7]
		df['t_loss'] = vals[:,8]
		df['gen'] = vals[:,11]
		df['stim'] = vals[:,12]

		#### from "play_script2.py"...
		for k, name in enumerate(names):
			df['lifetime'].append(fams['duration'][k])
			df['fate'].append(fams['fate'][k])
			ii = name[::-1].find('-')
			# print(name, ii)
			if ii != -1:  # format 1-a1-nnn
				df['well'].append(name[:-ii])
				df['relation'].append(name[-ii:])
				df['clone'].append(-1)
			else:  # format nnnnn010101
				df['well'].append(name[:5])
				df['relation'].append(name[5:])
				df['clone'].append(-1)
		lab, max_gen = 0, int(max(df['gen']))
		for igen in range(max_gen+1):
			for k in range(fams_size):
				cell_gen = df['gen'][k]
				if not cell_gen and not igen:
					df['clone'][k] = lab
					lab = lab + 1
				elif cell_gen and igen:
					parent_id = fams['mother'][k] - 1
					df['clone'][k] = df['clone'][parent_id]

		df_exp[path[7:]] = pd.DataFrame(df)
		keys.append(path[7:])

	# Clonally collapse all clones (ignore lost cells)
	df_CC = {exp: 
		{cond: {} for cond in np.unique(df_exp[exp]['stim'])} for exp in keys
	}
	for exp in keys:
		conds = np.unique(df_exp[exp]['stim'])
		for cond in conds:
			df = df_exp[exp][df_exp[exp].stim==cond]  # select all data == cond
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
					if len(df_gen) != 0.:  # check if data is empty at igen
						# collect division times
						t_div = np.array(df_gen[~np.isnan(df_gen['t_div'])]['t_div'])
						if len(t_div) != 0.:
							mean_t_div = np.mean(t_div)
							collapse[f't_div_{igen}'].append(mean_t_div)
						else:  # no division time information
							collapse[f't_div_{igen}'].append(np.nan)

						# collect death times
						t_death = np.array(df_gen[~np.isnan(df_gen['t_death'])]['t_death'])
						if len(t_death) != 0.:
							mean_t_death = np.mean(t_death)
							collapse[f't_death_{igen}'].append(mean_t_death)
						else:  # no death time information
							collapse[f't_death_{igen}'].append(np.nan)
					else:
						collapse[f't_div_{igen}'].append(np.nan)
						collapse[f't_death_{igen}'].append(np.nan)
			df_CC[exp][cond] = pd.DataFrame(collapse)
			df_CC[exp][cond] = df_CC[exp][cond].dropna(how='all')
	return (keys, df_exp, df_CC)

def filter_data(df_exp, exps):
	df_F = {exp: 
		{cond: {} for cond in np.unique(df_exp[exp]['stim'])} for exp in exps
	}
	df_F_CC = {exp: 
		{cond: {} for cond in np.unique(df_exp[exp]['stim'])} for exp in exps
	}
	for exp in exps:
		conds = np.unique(df_exp[exp]['stim'])
		for cond in conds:
			df = df_exp[exp][df_exp[exp]['stim']==cond].copy()
			
			max_gen = int(max(np.array(df['gen'])))
			collapse = {'clone': [], 'stim': []}
			for igen in range(max_gen+1):
				collapse['t_div_'+str(igen)] = []
				collapse['t_death_'+str(igen)] = []

			filtered_clone = []
			unique_clone = np.unique(df['clone'])
			for uc in unique_clone:
				clone = df[df['clone']==uc]

				### Filter on activated and ultimately non lost families
				lost_times = np.array(clone['t_loss'])
				death_times = np.array(clone['t_death'])
				div_times = np.array(clone['t_div'])

				number_lost = len(lost_times[~np.isnan(lost_times)])
				max_death_time = np.nanmax(death_times)
				max_lost_time = np.nanmax(lost_times)
				max_div_time = np.nanmax(div_times)
				
				# Filter the data according to:
				# 1. At least one division occurred
				# 2. Last observed event is death (not division or lost)
				if len(clone) > 1 and (
					(number_lost > 0 and max_death_time > max_lost_time) or
					(number_lost == 0 and max_death_time > max_div_time)):
					filtered_clone.append(uc)
					collapse['clone'].append(uc)
					collapse['stim'].append(cond)
					for igen in range(max_gen+1):
						df_gen = clone[clone['gen']==igen].copy()
						if len(df_gen) != 0:
							t_div = np.array(df_gen[~np.isnan(df_gen['t_div'])]['t_div'])
							if len(t_div) != 0:
								mean_t_div = np.mean(t_div)
								collapse['t_div_'+str(igen)].append(mean_t_div)
							else:
								collapse['t_div_'+str(igen)].append(np.nan)
							t_death = np.array(df_gen[~np.isnan(df_gen['t_death'])]['t_death'])
							if len(t_death) != 0:
								mean_t_death = np.mean(t_death)
								collapse['t_death_'+str(igen)].append(mean_t_death)
							else:
								collapse['t_death_'+str(igen)].append(np.nan)
						else:
							collapse['t_div_'+str(igen)].append(np.nan)
							collapse['t_death_'+str(igen)].append(np.nan)
			df_F[exp][cond] = df[df.clone.isin(filtered_clone)].copy()
			df_F_CC[exp][cond] = pd.DataFrame(collapse)
			df_F_CC[exp][cond] = df_F_CC[exp][cond].dropna(how='all')
	return df_F, df_F_CC

def save_dataframes(exps, df_exp, df_CC, df_F):
	for exp in exps:
		file_name = exp.replace('/','-')
		excel_wrter = pd.ExcelWriter(f'out/data/_processed/_parse/{file_name}.xlsx', engine='openpyxl')
		df_exp[exp].to_excel(excel_wrter, sheet_name='raw')

		conds = np.unique(df_exp[exp]['stim'])
		for cnd in conds:
			df_F[exp][cnd].to_excel(excel_wrter, sheet_name=f'Filtered_{cnd}')
			df_CC[exp][cnd].to_excel(excel_wrter, sheet_name=f'{cnd}')
			excel_wrter.save()
		excel_wrter.close()

def save_cc_times(exps, df_exp):
	for exp in exps:
		exp_lab = exp.replace('/', '_')
		# conds = np.unique(df_exp[exp]['stim'])
		conds = np.unique(list(df_exp[exp].keys()))  # find unique conditions
		for cond in conds:
			df = df_exp[exp][cond].copy()  # filtered data

			CONDS = []; CONDS.append(cond)
			TIME_TO_FIRST_DIV = []
			AVG_SUB_DIV = []
			TIME_TO_LAST_DIV = []
			TIME_TO_DEATH = []

			clones = np.unique(df['clone'])
			for cl in clones:
				tdie = np.mean(df[(df.clone==cl) & (df.fate=='died')].t_death.to_numpy())
				tdiv0 = df[(df.clone==cl) & (df.fate=='divided') & (df.gen==0)].t_div.to_numpy() # time to first division
				tdiv = np.mean(df[(df.clone==cl) & (df.fate=='divided') & (df.gen>0)].lifetime.to_numpy())  # average subsequent division time

				# NOTE: surrogate of division destiny times. It is problematic that, by my definition, Tld = Tdiv0 + Sub.Div. Which means that we won't see Tld < Tdiv0 case. Also, it's self circulating stupid logic that I've defined it to be correlated by definition, and determining its correlation coefficient. Probably, this is why as a proxy to measure Tdd, Giulio previously used Quiescence duration.
				cell_rel = df[(df.clone==cl) & (df.fate=='died')].relation.to_numpy()
				cell_rel_prec = [label[:-1] for label in cell_rel]  # determine the relation in the family tree
				y_clo = df[(df.clone==cl) & (df.fate=='divided') & (df.relation.isin(cell_rel_prec))].t_div.to_numpy()
				y_clo = y_clo[~np.isnan(y_clo)]
				tld = np.mean(y_clo)

				if len(tdiv0) > 0:
					TIME_TO_FIRST_DIV.append(tdiv0[0])
				else:
					TIME_TO_FIRST_DIV.append(np.nan)  # if no data put NaN
				AVG_SUB_DIV.append(tdiv)
				TIME_TO_DEATH.append(tdie)
				TIME_TO_LAST_DIV.append(tld)
			df_times = pd.DataFrame({
				"tdiv0": TIME_TO_FIRST_DIV,
				"tdiv": AVG_SUB_DIV,
				"tld": TIME_TO_LAST_DIV,
				"tdie": TIME_TO_DEATH
			})
			df_times.to_csv(f"./data/_processed/collapsed_times/{exp_lab}_{cond}.csv", index=False)


def rank_fam(df, fam):
	df_fam = df[df['clone']==fam]
	return max(np.ravel(df_fam[['t_birth', 't_death', 't_loss']]))

def rank_mean_fam(df, index):
	fam_vec = np.array(df.loc[index, :])
	fam_vec = [v for v in fam_vec if not np.isnan(v)]
	if not len(fam_vec): return -1.
	else: return max(fam_vec)

def ecdf(x):
	x = np.array(x)
	n = len(x)

	xs = np.sort(x)
	ys = np.arange(1, n+1)/float(n)
	return (xs, ys)

def mDD(v):
	vsize = len(v)
	
	num = np.sum([k * v[k] * float(np.power(2., -k)) for k in range(vsize)])
	den = np.sum([v[k] * float(np.power(2., -k)) for k in range(vsize)])
	if not den: return 0.0
	else: return num / den

def maxDD(v):
	vsize = len(v)
	return next(k for k in range(len(vsize))[::-1] if v[k] != 0.)  # iterate in reverse order

def minDD(v):
	vsize = len(v)
	return next(k for k in range(vsize) if v[k] != 0.)

def frayDD(v):
	vsize = len(v)
	return maxDD(v) - minDD(v)

### UTILITY
def check_possible(v):
	vsize = len(v)
	return np.sum([v[k]*float(np.power(2., -k)) for k in range(vsize)]) <= 1.

def check_complete(v):
	vsize = len(v)
	return np.sum([v[k]*float(np.power(2., -k)) for k in range(vsize)]) == 1.

def check_nonneg(v):
	return all(v >= 0.)


## Plot nice tick marks along the plot path: https://github.com/xnx/lineticks
# from matplotlib import transforms
# def get_perp_vec(u1, u2, direction=1):
# 	"""Return the unit vector perpendicular to the vector u2-u1."""

# 	x1, y1 = u1
# 	x2, y2 = u2
# 	vx, vy = x2-x1, y2-y1
# 	v = np.linalg.norm((vx, vy))
# 	wx, wy = -vy/v * direction, vx/v * direction
# 	return wx, wy

# def get_av_vec(u1, u2):
# 	"""Return the average unit vector between u1 and u2."""

# 	u1x, u1y = u1
# 	u2x, u2y = u2
# 	dx, dy = u1x + u2x, u1y + u2y
# 	dlen = np.linalg.norm((dx,dy))
# 	return dx/dlen, dy/dlen

# class LineTicks:
# 	def __init__(self, line, idx, tick_length, direction=1, label=None, **kwargs):
# 		self.line = line
# 		self.idx = idx
# 		self.tick_length = tick_length
# 		self.direction = direction
# 		self.label = label
# 		self.ticks = []
# 		self.tick_labels = []

# 		self.tick_styles = kwargs
# 		# If no colour is specified for the ticks, set it to the line colour
# 		if not set(('c', 'color')).intersection(kwargs.keys()):
# 			self.tick_styles['color'] = self.line.get_color()

# 		self.ax = line.axes
# 		self.ax.callbacks.connect('xlim_changed', self.on_change_lims)
# 		self.ax.callbacks.connect('ylim_changed', self.on_change_lims)
# 		cid = self.ax.figure.canvas.mpl_connect('resize_event',self.on_resize) 
# 		self.add_ticks(self.ax)

# 	def add_ticks(self, ax):
# 		ax.set_autoscale_on(False)  # Otherwise, infinite loop
# 		# Transform to  display coordinates
# 		z =ax.transData.transform(np.array(self.line.get_data()).T)
# 		x, y = zip(*z)

# 		# Remove existing ticks
# 		for tick in self.ticks:
# 			ax.lines.remove(tick)
# 		# Remove references to the ticks so they can be garbage-collected
# 		self.ticks = []

# 		# Remove any existing tick labels
# 		for ticklabel in self.tick_labels:
# 			ax.texts.remove(ticklabel)
# 		self.tick_labels = []

# 		for j,i in enumerate(self.idx):
# 			if i == 0:
# 				# The first tick is perpendicular to the line between the
# 				# first two points
# 				tx, ty = get_perp_vec((x[0], y[0]), (x[1], y[1]), self.direction)
# 			elif i == len(x)-1:
# 				# The last tick is perpendicular to the line between the
# 				# last two points
# 				tx, ty = get_perp_vec((x[-2], y[-2]), (x[-1], y[-1]), self.direction)
# 			else:
# 				# General tick marks bisect the incoming and outgoing line
# 				# segments
# 				u1 = get_perp_vec((x[i-1], y[i-1]), (x[i], y[i]), self.direction)
# 				u2 = get_perp_vec((x[i], y[i]), (x[i+1], y[i+1]), self.direction)
# 				tx, ty = get_av_vec(u1, u2)
# 			tx, ty = self.tick_length * tx, self.tick_length * ty
# 			this_tick, = ax.plot((x[i],x[i]+tx), (y[i],y[i]+ty), transform=transforms.IdentityTransform(), **self.tick_styles)
# 			self.ticks.append(this_tick)

# 			if self.label:
# 				this_ticklabel = ax.text(x[i]+tx*3, y[i]+ty*3, self.label[j], transform=transforms.IdentityTransform(), ha='center', va='center', clip_on=True)
# 				self.tick_labels.append(this_ticklabel)

# 	def on_change_lims(self, ax):
# 		self.add_ticks(ax)

# 	def on_resize(self, event):
# 		self.add_ticks(self.ax)

## Find index of nearest value in an array: https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
# def find_nearest(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return idx