"""
Last edit: 26-Feb-2019

Reshape an input data format to a 4D array (icnd, itpt, igen, irep)
"""
import numpy as np
import scipy.stats as sps
from .utils import remove_empty


def compute_total_cells(data, conditions, num_tps, gen_per_condition):
	"""
	All parameters are output of file_reader.py object, which consists meta information about the data itself.

	:param data: (nested list) number of cells per generation (4-dimensional array) = data[icnd][itpt][irep][igen]
	:param conditions: (list) names of condition
	:param num_tps: (list) number of time points per condition
	:param gen_per_condition: (list) number of maximum generations per condition
	:return: (tuple) [average total cells, total cells with replicates, standard error of means]
	"""
	num_conditions = len(conditions)

	# this loop computes average total cells
	max_length = 0
	total_cells = [[] for _ in range(num_conditions)]
	for icnd in range(num_conditions):
		for itpt in range(num_tps[icnd]):
			cell = 0.
			for igen in range(gen_per_condition[icnd]+1):
				temp = 0.
				replicate = 0.

				size_of_data = len(data[icnd][itpt][igen])
				# check for single replicate and update
				if size_of_data == 0:
					replicate = 1.
				# loop through replicates
				for datum in data[icnd][itpt][igen]:
					if datum is not None:
						temp += datum
						replicate += 1
					# this finds maximum number of replicates in the experiment (useful for asymmetric data)
					if max_length < size_of_data:
						max_length = size_of_data
				temp = temp / replicate
				cell += temp
			total_cells[icnd].append(cell)

	filtered_total_cells = remove_empty(total_cells)

	# this loop computes total cells for EACH replicates
	total_cells_reps = [[] for _ in range(num_conditions)]
	total_cells_reps2 = [[[] for _ in range(max(num_tps))] for _ in range(num_conditions)]
	for icnd in range(num_conditions):
		for itpt in range(num_tps[icnd]):
			tmp = [0 for _ in range(max_length)]
			for igen in range(gen_per_condition[icnd]+1):
				for irep, datum in enumerate(data[icnd][itpt][igen]):
					tmp[irep] += datum
			total_cells_reps[icnd].append(tmp)
			for idx in range(len(data[icnd][itpt][igen])):
				total_cells_reps2[icnd][itpt].append(tmp[idx])

	filtered_total_cells_reps = remove_empty(total_cells_reps)
	filtered_total_cells_reps2 = remove_empty(total_cells_reps2)

	# compute standard error of mean for replicates
	total_cells_sem = [[] for _ in range(num_conditions)]
	for icnd in range(num_conditions):
		for itpt in range(num_tps[icnd]):
			total_cells_sem[icnd].append(sps.sem(filtered_total_cells_reps2[icnd][itpt]))

	filtered_total_cells_sem = remove_empty(total_cells_sem)

	return filtered_total_cells, filtered_total_cells_reps, filtered_total_cells_sem


def total_cohorts(data, conditions, num_tps, gen_per_condition):
	num_conditions = len(conditions)

	max_length = 0
	total_cohort = [[] for _ in range(num_conditions)]
	total_cohort_norm0 = [[] for _ in range(num_conditions)]
	total_cohort_norm1 = [[] for _ in range(num_conditions)]
	for icnd in range(num_conditions):
		for itpt in range(num_tps[icnd]):
			cohort, cohor1, cohort2 = 0., 0., 0.
			for igen in range(gen_per_condition[icnd]+1):
				temp = 0.
				replicate = 0.
				size_of_data = len(data[icnd][itpt][igen])
				if size_of_data == 0:
					replicate = 1.
				for datum in data[icnd][itpt][igen]:
					if datum is not None:
						temp += datum * np.power(2., float(-igen))
						replicate += 1
					if max_length < size_of_data:
						max_length = size_of_data
				temp = temp / replicate
				cohort += temp
			total_cohort[icnd].append(cohort)
			if itpt == 0:
				ref1 = cohort
			if itpt == 1:
				ref2 = cohort
			cohort1 = cohort / ref1
			total_cohort_norm0[icnd].append(cohort1)
			if itpt > 0:
				cohort2 = cohort / ref2
				total_cohort_norm1[icnd].append(cohort2)

	filtered_total_cohort = remove_empty(total_cohort)
	filtered_total_cohort_norm0 = remove_empty(total_cohort_norm0)
	filtered_total_cohort_norm1 = remove_empty(total_cohort_norm1)

	# this loop computes total cells for EACH replicates
	total_cohort_reps = [[] for _ in range(num_conditions)]
	total_cohort_reps2 = [[[] for _ in range(max(num_tps))] for _ in range(num_conditions)]
	total_cohort_reps_norm0 = [[] for _ in range(num_conditions)]
	total_cohort_reps2_norm0 = [[[] for _ in range(max(num_tps))] for _ in range(num_conditions)]
	total_cohort_reps_norm1 = [[] for _ in range(num_conditions)]
	total_cohort_reps2_norm1 = [[[] for _ in range(max(num_tps)-1)] for _ in range(num_conditions)]
	for icnd in range(num_conditions):
		for itpt in range(num_tps[icnd]):
			tmp = [0 for _ in range(max_length)]
			tmp2 = [0 for _ in range(max_length)]
			tmp3 = [0 for _ in range(max_length)]
			for igen in range(gen_per_condition[icnd]+1):
				for irep, datum in enumerate(data[icnd][itpt][igen]):
					tmp[irep] += datum * np.power(2., float(-igen))

				# normalise with itpt=0
				if itpt == 0:
					# zeros = [i for i, e in enumerate(tmp) if e == 0]
					ref = tmp
				for irep in range(len(data[icnd][itpt][igen])):
					# if irep in zeros:
					# 	tmp2[irep] = np.NaN
					tmp2[irep] = tmp[irep] / ref[irep]				
				# normalise with itpt=1
				if itpt == 1:
					ref2 = tmp
				if itpt > 0:
					for irep in range(len(data[icnd][itpt][igen])):
						tmp3[irep] = tmp[irep] / ref2[irep]
			total_cohort_reps[icnd].append(tmp)
			total_cohort_reps_norm0[icnd].append(tmp2)
			if itpt > 0:
				total_cohort_reps_norm1[icnd].append(tmp3)
			for irep in range(len(data[icnd][itpt][igen])):
				total_cohort_reps2[icnd][itpt].append(tmp[irep])
				total_cohort_reps2_norm0[icnd][itpt].append(tmp2[irep])
				total_cohort_reps2_norm1[icnd][itpt-1].append(tmp3[irep])
	filtered_total_cohort_reps = remove_empty(total_cohort_reps)
	filtered_total_cohort_reps2 = remove_empty(total_cohort_reps2)

	filtered_total_cohort_reps_norm0 = remove_empty(total_cohort_reps_norm0)
	filtered_total_cohort_reps2_norm0 = remove_empty(total_cohort_reps2_norm0)

	filtered_total_cohort_reps_norm1 = remove_empty(total_cohort_reps_norm1)
	filtered_total_cohort_reps2_norm1 = remove_empty(total_cohort_reps2_norm1)

	# compute standard error of mean for replicates
	total_cohort_sem = [[] for _ in range(num_conditions)]
	total_cohort_sem_norm0 = [[] for _ in range(num_conditions)]
	total_cohort_sem_norm1 = [[] for _ in range(num_conditions)]
	for icnd in range(num_conditions):
		for itpt in range(num_tps[icnd]):
			total_cohort_sem[icnd].append(sps.sem(filtered_total_cohort_reps2[icnd][itpt]))
			total_cohort_sem_norm0[icnd].append(sps.sem(filtered_total_cohort_reps2_norm0[icnd][itpt]))
			if itpt > 0:
				total_cohort_sem_norm1[icnd].append(sps.sem(filtered_total_cohort_reps2_norm1[icnd][itpt-1]))

	filtered_total_cohort_sem = remove_empty(total_cohort_sem)
	filtered_total_cohort_sem_norm0 = remove_empty(total_cohort_sem_norm0)
	filtered_total_cohort_sem_norm1 = remove_empty(total_cohort_sem_norm1)
	return filtered_total_cohort, filtered_total_cohort_reps, filtered_total_cohort_sem, filtered_total_cohort_norm0, filtered_total_cohort_reps_norm0, filtered_total_cohort_sem_norm0, filtered_total_cohort_norm1, filtered_total_cohort_reps_norm1, filtered_total_cohort_sem_norm1, 


def cohorts(data, conditions, num_tps, gen_per_condition):
	num_conditions = len(conditions)
	
	max_length = 0
	cohort_gens = [[] for _ in range(num_conditions)]
	for icnd in range(num_conditions):
		for itpt in range(num_tps[icnd]):
			cohort_arr = []
			for igen in range(gen_per_condition[icnd]+1):
				cohort, replicate = 0., 0.

				size_of_data = len(data[icnd][itpt][igen])
				if size_of_data == 0:
					replicate = 1.
				
				for datum in data[icnd][itpt][igen]:
					if datum is not None:
						cohort += datum / (2.**igen)
						replicate += 1
					if max_length < size_of_data:
						max_length = size_of_data
				cohort = cohort / replicate

				cohort_arr.append(cohort)
			cohort_gens[icnd].append(cohort_arr)
	
	filtered_cohort_gens = remove_empty(cohort_gens)

	cohort_gens_reps = [[[] for _ in range(max(num_tps))] for _ in range(num_conditions)]
	for icnd in range(num_conditions):
		for itpt in range(num_tps[icnd]):
			tmp = [[] for _ in range(max_length)]
			for igen in range(gen_per_condition[icnd]+1):
				for irep, datum in enumerate(data[icnd][itpt][igen]):
					tmp[irep].append(datum/(2.**igen))
			for idx in range(len(data[icnd][itpt][igen])):
				cohort_gens_reps[icnd][itpt].append(tmp[idx])
	
	filtered_cohort_gens_reps = remove_empty(cohort_gens_reps)

	# resort filtered dataset to compute SEM : schema - [icnd][itpt][igen][irep]
	resorted_data = [
		[
			[
				[] for _ in range(max(gen_per_condition)+1)
			] for _ in range(max(num_tps))
		] for _ in range(num_conditions)
	]
	cohort_gens_sem = [
		[] for _ in range(num_conditions)
	]
	for icnd in range(len(filtered_cohort_gens_reps)):
		for itpt in range(len(filtered_cohort_gens_reps[icnd])):
			for irep, l in enumerate(filtered_cohort_gens_reps[icnd][itpt]):
				for igen, datum in enumerate(l):
					resorted_data[icnd][itpt][igen].append(datum)
			tmp = []
			for idx in range(len(resorted_data[icnd][itpt])):
				tmp.append(sps.sem(resorted_data[icnd][itpt][idx]))
			cohort_gens_sem[icnd].append(tmp)

	filtered_cohort_gens_sem = remove_empty(cohort_gens_sem)

	return filtered_cohort_gens, filtered_cohort_gens_reps, filtered_cohort_gens_sem


def sort_cell_generations(data, conditions, num_tps, gen_per_condition):
	"""
	This function organises cell-generation profile.

	:param data: (nested list) number of cells per generation (4-dimensional array) = data[icnd][itpt][irep][igen]
	:param conditions: (list) names of condition
	:param num_tps: (list) number of time points per condition
	:param gen_per_condition: (list) number of maximum generations per condition
	:return: (tuple) [average cell per gen, cell per gen with replicates, standard error of means]
	"""
	num_conditions = len(conditions)

	# this loop computes average cell numbers : dynamically determines replicates
	max_length = 0
	cell_gens = [[] for _ in range(num_conditions)]
	for icnd in range(num_conditions):
		for itpt in range(num_tps[icnd]):
			gen_arr = []
			for igen in range(gen_per_condition[icnd]+1):
				cell = 0.
				replicate = 0.

				size_of_data = len(data[icnd][itpt][igen])
				if size_of_data == 0:
					replicate = 1.

				for datum in data[icnd][itpt][igen]:
					if datum is not None:
						cell += datum
						replicate += 1.
					if max_length < size_of_data:
						max_length = size_of_data
				cell = cell / replicate

				gen_arr.append(cell)
			cell_gens[icnd].append(gen_arr)

	filtered_cell_gens = remove_empty(cell_gens)

	cell_gens_reps = [[[] for _ in range(max(num_tps))] for _ in range(num_conditions)]
	for icnd in range(num_conditions):
		for itpt in range(num_tps[icnd]):
			tmp = [[] for _ in range(max_length)]
			for igen in range(gen_per_condition[icnd]+1):
				for irep, datum in enumerate(data[icnd][itpt][igen]):
					tmp[irep].append(datum)
			for idx in range(len(data[icnd][itpt][igen])):
				cell_gens_reps[icnd][itpt].append(tmp[idx])

	filtered_cell_gens_reps = remove_empty(cell_gens_reps)

	# resort filtered dataset to compute SEM : schema - [icnd][itpt][igen][irep]
	resorted_data = [
		[
			[
				[] for _ in range(max(gen_per_condition)+1)
			] for _ in range(max(num_tps))
		] for _ in range(num_conditions)
	]
	cell_gens_sem = [
		[] for _ in range(num_conditions)
	]
	for icnd in range(len(filtered_cell_gens_reps)):
		for itpt in range(len(filtered_cell_gens_reps[icnd])):
			for irep, l in enumerate(filtered_cell_gens_reps[icnd][itpt]):
				for igen, datum in enumerate(l):
					resorted_data[icnd][itpt][igen].append(datum)
			tmp = []
			for idx in range(len(resorted_data[icnd][itpt])):
				tmp.append(sps.sem(resorted_data[icnd][itpt][idx]))
			cell_gens_sem[icnd].append(tmp)

	filtered_cell_gens_sem = remove_empty(cell_gens_sem)

	return filtered_cell_gens, filtered_cell_gens_reps, filtered_cell_gens_sem


