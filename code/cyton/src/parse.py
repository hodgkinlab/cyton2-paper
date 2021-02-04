import os, re
import numpy as np
from ._reader import ReadData
from ._manager import compute_total_cells, sort_cell_generations, cohorts


def parse_data(path, target_data):
	"""
	Take target data files and rearrange them into a single nested dictionary data object

	:param path: (str) path to the target data folder
	:param target_data: (list) list of data file names
	:return: (dict) a dictionary with file names as key. each key has meta information, total cells, and cells per generation
	"""
	df = {}  # data frame contains all parsed data organised by file name
	for raw_data in target_data:
		fname = os.path.splitext(raw_data)[0]
		df[fname] = {
			'reader': None,
			'cells': {'avg': None, 'rep': None, 'sem': None},
			'cgens': {'avg': None, 'rep': None, 'sem': None},
			'cohorts_gens': {'avg': None, 'rep': None, 'sem': None}
		}

		full_path = os.path.join(path, raw_data)

		reader = ReadData(full_path)
		cells = compute_total_cells(reader.data, reader.condition_names, reader.num_time_points, reader.generation_per_condition)
		cgens = sort_cell_generations(reader.data, reader.condition_names, reader.num_time_points, reader.generation_per_condition)
		cohorts_gens = cohorts(reader.data, reader.condition_names, reader.num_time_points, reader.generation_per_condition)

		df[fname]['reader'] = reader  # read individual data and store all information in one array

		df[fname]['cells']['avg'] = cells[0]  # average total cell numbers
		df[fname]['cells']['rep'] = cells[1]  # total cell numbers per replicate
		df[fname]['cells']['sem'] = cells[2]  # SEM of total cell numbers (calculated from reps)

		df[fname]['cgens']['avg'] = cgens[0]  # average cell numbers per generation
		df[fname]['cgens']['rep'] = cgens[1]  # cell numbers per generation per replicates
		df[fname]['cgens']['sem'] = cgens[2]  # SEM of cell number per generation

		df[fname]['cohorts_gens']['avg'] = cohorts_gens[0]  # average cohort numbers per generation per time point
		df[fname]['cohorts_gens']['rep'] = cohorts_gens[1]  # cohort numbers per generation per time point per replicate
		df[fname]['cohorts_gens']['sem'] = cohorts_gens[2]  # SEM of cohorts per generation per time point
	return df


def concatenate(df):
	"""
	Concatenate parsed data. Following operations are handled,
		1. Gather total harvested times from each of dataset
		2. Restructure the data in ascending order of harvested times for raw cell numbers
		3. Normalise the data via total cohort normalisation
			- NB: Not all dataset have symmetric harvested times points, which makes some time points have more
			replicates than the other. Therefore, take the average of the total cohort per time point and apply
			normalisation with respect to the average total cohort.
		4. Interlace the normalised data points in ascending order of harvested times
	:param df:
	:return:
	"""
	conds = []
	buffer_ht = {}
	concat_df = {}
	# iterate over files in data frame to initialse the concatenation per conditions
	for key, sub in df.items():
		# reorganise harvested times
		for icnd, condition in enumerate(sub['reader'].condition_names):
			dose = [float(s) for s in re.findall(r'-?\d+\.?\d*', condition)]
			unit = [s for s in re.findall(r'[a-zA-Z][a-zA-Z]-?\d\d|(?:[a-zA-Z][a-zA-Z])', condition)]  # returns [0] experiment id
			c = dose[0]
			conds.append(c)  # list of conditions for all data files
	conds = sorted(list(set(conds)), key=float)  # remove duplicates and sort in ascending order ([0-9] [a-z])
	for idx, c in enumerate(conds):
		if len(unit) > 1:
			composite_unit = str(conds[idx]) + ' ' + unit[0] + '/' + unit[1]
		else:
			composite_unit = str(conds[idx]) + ' ' + unit[0]
		conds[idx] = composite_unit
		buffer_ht[str(conds[idx])] = []  # define empty list for all possible conditions
		concat_df[str(conds[idx])] = {}

	# concat_df['conditions'] = conds
	for key, sub in df.items():
		for icnd, c in enumerate(sub['reader'].condition_names):
			for ht in sub['reader'].harvested_times[icnd]:
				buffer_ht[c].append(ht)
			buffer_ht[c] = sorted(list(set(buffer_ht[c])))  # remove duplicates and sort in ascending order
			concat_df[c]['ht'] = buffer_ht[c]  # store in concatenated data frame

			# define sub categories
			concat_df[c]['gens'] = [[] for _ in range(len(buffer_ht[c]))]
			concat_df[c]['num_reps'] = [[] for _ in range(len(buffer_ht[c]))]
			concat_df[c]['cgens'] = {}
			concat_df[c]['cgens']['avg'] = [[] for _ in range(len(buffer_ht[c]))]
			concat_df[c]['cgens']['rep'] = [[] for _ in range(len(buffer_ht[c]))]
			concat_df[c]['cgens']['normed'] = [[] for _ in range(len(buffer_ht[c]))]
			concat_df[c]['x'] = []  # final x data
			concat_df[c]['y'] = []  # final y data
	
	# loop through cell number data for reorganise correspond to harvested time points
	for key, sub in df.items():
		for icnd, c in enumerate(sub['reader'].condition_names):
			concent1 = float(re.findall(r'-?\d+\.?\d*', c)[0])
			concent2 = float(re.findall(r'-?\d+\.?\d*', sub['reader'].condition_names[icnd])[0])
			num_reps = [len(l) for l in sub['cgens']['rep'][icnd]]

			# concatenate raw data numbers in ascending order of harvested time
			for j, ht in enumerate(buffer_ht[c]):
				if ht in sub['reader'].harvested_times[icnd] and concent1 == concent2:
					itpt = sub['reader'].harvested_times[icnd].index(ht)
					buffer_cells = []
					
					for irep in range(num_reps[itpt]):
						buffer_gens, cells = [], []
						for igen in range(sub['reader'].generation_per_condition[icnd]+1):
							buffer_gens.append(igen)
							cells.append(sub['cgens']['rep'][icnd][itpt][irep][igen])  # gather cell numbers
						buffer_cells.append(cells)  # attach all cells per replicate
						concat_df[c]['gens'][j].append(buffer_gens)
						concat_df[c]['cgens']['rep'][j].append(cells)
					concat_df[c]['num_reps'][j].append(num_reps[itpt])
					concat_df[c]['cgens']['avg'][j].append(np.average(buffer_cells, axis=0))  # over per data file

	# normalisation
	for c, sub in concat_df.items():
		total_cohorts = 0.
		for itpt, cgens in enumerate(sub['cgens']['rep']):
			cohort_sum = 0.
			for cgens_rep in cgens:
				for igen, cell in enumerate(cgens_rep):
					cohort_sum += cell * np.power(2., -igen)
			cohort_sum = cohort_sum/np.sum(sub['num_reps'][itpt])
			total_cohorts += cohort_sum
		for itpt, cgens in enumerate(sub['cgens']['rep']):
			for cgens_rep in cgens:
				buffer_normed = []
				for igen, cell in enumerate(cgens_rep):
					normed = cell * np.power(2., -igen)/total_cohorts
					concat_df[c]['x'].append(igen)
					concat_df[c]['y'].append(normed)
					buffer_normed.append(normed)
				concat_df[c]['cgens']['normed'][itpt].append(buffer_normed)
	return concat_df
