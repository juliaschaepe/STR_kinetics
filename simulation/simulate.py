import numpy as np
import argparse
import tqdm
import matplotlib.pyplot as plt

'''
This file contains functions to run a gillespie simulation for TF search with or 
without repeats flanking a motif. Sensitivity analysis functions are also included.
The simulation and the sensitivity analysis functions are intended to be parallelized
to speed up computation. See Gillespie method for information about states and 
equations defining rates, affinities, and reaction likelihoods.

Note that here numbers are used to represent different states in the model. 
1 = nucleus, 2 = local, 3 = motif bound, 4 = flanks bound
'''

# Global variables
DT_INDEX = 0
TIME_INDEX = 1
NUCLEUS_INDEX = 2
LOCAL_INDEX = 3
MOTIF_INDEX = 4
FLANK_INDEX = 5
RXN_INDEX = 6
SIM_TIME = 5e3
MAX_TIME = 1e5
REPEAT_FACTOR = 0.1
RANDOM_FACTOR = 100

# parses input and runs appropriate simulation or sensitivity analysis
def main():
	parser = argparse.ArgumentParser(description='Get run number and sensitivity analysis target')
	parser.add_argument('run_num', type=str, help='run number')
	parser.add_argument('target', type=str, help='sensitivity analysis target variable')
	parser.add_argument("-y0", type=int, nargs="+", default=[1, 0, 0, 0, 1, 100])
	args = parser.parse_args()
	factors = np.geomspace(1e-3, 1, num=10)

	# these targets are all set up for parallelization
	if args.target == 'n_flanks':
		n_flanks_sensitivity(args.target, args.run_num, args.y0, factors)
	if args.target == 'core_affinity':
		core_affinity_sensitivity(args.target, args.run_num, args.y0, factors)
	if args.target == 'koff_slope':
		koff_slope_sensitivity(args.target, args.run_num, args.y0, factors)
	if args.target == 'n_TF':
		n_tf_sensitivity(args.target, args.run_num, args.y0, factors)
	if args.target == 'switching_rate':
		switching_rate_sensitivity(args.target, args.run_num, args.y0, factors)
	if args.target == 'diffusion':
		diffusion_sensitivity(args.target, args.run_num, args.y0, factors)
	if args.target == 'DNA_concentration':
		dna_concentration_sensitivity(args.target, args.run_num, args.y0, factors)
	if args.target == 'simulation':
		factors = np.geomspace(1e-4, 10)
		simulation(factors, args.run_num, args.y0)

	# these targets are set up to run locally for quick testing
	if args.target == 'one_simulation':
		one_simulation(args.y0)
	if args.target == 'mfpt_simulation':
		mfpt_simulation(args.run_num, args.y0)

	print('Run completed')


# returns kinetic parameters for gillespie simulation
def get_k_array(n_flanks, factor, core_affinity=1e-7, switching_rate=0.5,
				tf_diffusion=1, koff_slope=0.203, koff_intercept=0.414,
				local_vol=1e-4):
	nuc_vol = 3 # um^3
	D = tf_diffusion  # um^2/s
	R = np.cbrt(3 * local_vol / (4 * np.pi)) # um

	Kd_23 = core_affinity  # source: MITOMI data
	Kd_24 = core_affinity / factor
	Kd_34 = Kd_24 / Kd_23
	Kd_12 = nuc_vol / local_vol

	k12 = 4 * np.pi * D * R / nuc_vol # smoluchowski diffusion, 1/s
	k21 = k12 * Kd_12 # to maintain "detailed balance", 1/s

	k32 = np.exp(np.log(Kd_23)*koff_slope + koff_intercept) # source: MITOMI data - this is koff, 1/s
	k23 = k32 / Kd_23  # detailed balance - this is kon, 1/Ms

	k42 = np.exp(np.log(Kd_24)*koff_slope + koff_intercept)  # source: MITOMI data - this is koff, 1/s
	k24 = k42 / Kd_24  # detailed balance - this is kon, 1/Ms

	# k43 = switching_rate
	# k34 = k43 / Kd_34  # detailed balance, 1/s

	k34 = Kd_24 * k24
	k43 = Kd_34 * k34

	return k12, k21, k23, k24, k32, k34, k42, k43


# Gillespie simulation of TFsearch
def simulate_tf_search(sim_T, max_T, y0, k_array, DNA=5e-5, mfpt_only=False):
	# initialization
	k12, k21, k23, k24, k32, k34, k42, k43 = k_array
	stored_data = False
	first_passage = False
	first_passage_time = -1
	rxn = -1
	t = 0
	i = 1
	y = y0.copy()
	n_rows = 100000
	sim_data = np.zeros((n_rows, len(y0) + 3))
	sim_data[0] = np.hstack([0, t, y, rxn])

	# maps a given reaction to how molecules should be updated (subtract from, add to)
	w_mapping = [([0], [1]),	# diffusion into local volume
				 ([1], [0]),	# diffusion out of local volume
				 ([1, 4], [2]),	# binding to motif
				 ([1, 5], [3]),	# binding to flanks
				 ([2], [1, 4]),	# unbinding from motif
				 ([2, 5], [3, 4]),	# switching onto flanks from motif
				 ([3], [1, 5]),	# unbinding from flanks
				 ([3, 4], [2, 5])]	# switching onto motif from flanks

	while t < max_T:

		# calculates likelihood of each reaction
		w12 = k12 * y[0]
		w21 = k21 * y[1]
		w23 = k23 * y[1] * y[4] * DNA
		w32 = k32 * y[2]
		w24 = k24 * y[1] * y[5] * DNA
		w42 = k42 * y[3]
		w34 = k34 * y[2]
		w43 = k43 * y[3]

		# ensures conservation of 1 motif and 100 flanks in the case that there is more than one TF
		if y[4] == 0:
			w43 = 0
		if y[5] == 0:
			w34 = 0

		# calculates tau
		w_arr = [w12, w21, w23, w24, w32, w34, w42, w43]
		W = np.sum(w_arr)
		tau = -np.log(np.random.rand()) / W
		t = t + tau

		# chooses next reaction
		rand = np.random.rand()
		for j in range(len(w_arr)):
			if rand <= (np.sum(w_arr[:j+1]) / W):
				# if j in [2, 7] and not first_passage:
				rxn = j
				idx_from, idx_to = w_mapping[j]
				for idx in idx_from:
					y[idx] -= 1
				for idx in idx_to:
					y[idx] += 1
				break

		# checks to see if motif is bound for the first time
		if y[4] == 0 and not first_passage:
			first_passage_time = t
			first_passage = True

		# allocates more space if needed so that sim_data is not stored dynamically
		if i >= n_rows:
			sim_data = np.vstack((sim_data, np.zeros((n_rows, len(y0) + 3))))
			n_rows += n_rows

		# updates sim_data
		sim_data[i] = np.hstack([tau, t, y, rxn])
		i += 1

		# stores data for the amount of simulation time (for calculating occupancy)
		if t >= sim_T and not stored_data:
			sim_data_occ = np.asarray(sim_data)
			sim_data_occ = sim_data_occ[:np.argmax(sim_data_occ[:, 1]) + 1, :]
			stored_data = True

		# returns if only mfpt is needed
		if mfpt_only and first_passage:
			sim_data_occ = np.asarray(sim_data)
			sim_data_occ = sim_data_occ[:np.argmax(sim_data_occ[:, 1]) + 1, :]
			return sim_data_occ, first_passage_time

		# returns if mfpt and simulation data is stored
		if stored_data and first_passage:
			return sim_data_occ, first_passage_time

	# returns an MFPT 10 times the maximum if it did not occur within maximum time
	return sim_data_occ, max_T*10


# sensitivity analysis for number of flanks
def n_flanks_sensitivity(target, run_num, y0, factors):
	n_flanks = get_run_vars(target)
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(
		len(factors), len(n_flanks))
	for i, factor in enumerate(factors):
		for j, n in enumerate(n_flanks):
			y0[-1] = n
			k_array = get_k_array(n, factor)
			sim_data, first_passage_time = simulate_tf_search(SIM_TIME, MAX_TIME, y0, k_array)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, FLANK_INDEX)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX,
																FLANK_INDEX)
	save_output(target, run_num, first_passage, mean_occupancy_mot,
				mean_occupancy_flanks, mean_occupancy_local)


# sensitivity analysis for motif affinity
def core_affinity_sensitivity(target, run_num, y0, factors):
	core_affinities = get_run_vars(target)
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(
		len(factors), len(core_affinities))
	for i, factor in enumerate(factors):
		for j, Kd in enumerate(core_affinities):
			k_array = get_k_array(y0[-1], factor, core_affinity=Kd)
			sim_data, first_passage_time = simulate_tf_search(SIM_TIME, MAX_TIME, y0, k_array)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, FLANK_INDEX)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX,
																FLANK_INDEX)
	save_output(target, run_num, first_passage, mean_occupancy_mot,
				mean_occupancy_flanks, mean_occupancy_local)


# sensitivity analysis for koff slope
def koff_slope_sensitivity(target, run_num, y0, factors):
	koff_slopes = get_run_vars(target)
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(
		len(factors), len(koff_slopes))
	for i, factor in enumerate(factors):
		for j, koff_slope in enumerate(koff_slopes):
			k_array = get_k_array(y0[-1], factor, koff_slope=koff_slope)
			sim_data, first_passage_time = simulate_tf_search(SIM_TIME, MAX_TIME, y0, k_array)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, FLANK_INDEX)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX,
																FLANK_INDEX)
	save_output(target, run_num, first_passage, mean_occupancy_mot,
				mean_occupancy_flanks, mean_occupancy_local)


# sensitivity analysis for koff intercept
def koff_intercept_sensitivity(target, run_num, y0, factors):
	koff_intercepts = get_run_vars(target)
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(
		len(factors), len(koff_intercepts))
	for i, factor in enumerate(factors):
		for j, koff_intercept in enumerate(koff_intercepts):
			k_array = get_k_array(y0[-1], factor, koff_intercept=koff_intercept)
			sim_data, first_passage_time = simulate_tf_search(SIM_TIME, MAX_TIME, y0, k_array)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, FLANK_INDEX)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX,
																FLANK_INDEX)
	save_output(target, run_num, first_passage, mean_occupancy_mot,
				mean_occupancy_flanks, mean_occupancy_local)


# sensitivity analysis for number of TFs
def n_tf_sensitivity(target, run_num, y0, factors):
	TF_numbers = get_run_vars(target)
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(
		len(factors), len(TF_numbers))
	for i, factor in enumerate(factors):
		for j, n_TF in enumerate(TF_numbers):
			k_array = get_k_array(y0[-1], factor)
			y0[0] = n_TF
			sim_data, first_passage_time = simulate_tf_search(SIM_TIME, MAX_TIME, y0, k_array)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, FLANK_INDEX)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX,
																FLANK_INDEX)
	save_output(target, run_num, first_passage, mean_occupancy_mot, mean_occupancy_flanks,
					mean_occupancy_local)


# sensitivity analysis for flanks-->motif switching rate
def switching_rate_sensitivity(target, run_num, y0, factors):
	switching_kinetics = get_run_vars(target)
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(
		len(factors), len(switching_kinetics))

	for i, factor in enumerate(factors):
		print(factor)
		for j, switching_rate in enumerate(switching_kinetics):
			k_array = get_k_array(y0[-1], factor, switching_rate=switching_rate)
			sim_data, first_passage_time = simulate_tf_search(SIM_TIME, MAX_TIME, y0, k_array)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, FLANK_INDEX)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX,
																FLANK_INDEX)
	save_output(target, run_num, first_passage, mean_occupancy_mot, mean_occupancy_flanks,
					mean_occupancy_local)


# sensitivity analysis for TF diffusion constant
def diffusion_sensitivity(target, run_num, y0, factors):
	diffusion_constants = get_run_vars(target)
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(
		len(factors), len(diffusion_constants))
	for i, factor in enumerate(factors):
		for j, diffusion_constant in enumerate(diffusion_constants):
			k_array = get_k_array(y0[-1], factor, tf_diffusion=diffusion_constant)
			sim_data, first_passage_time = simulate_tf_search(SIM_TIME, MAX_TIME, y0, k_array)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, FLANK_INDEX)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX,
																FLANK_INDEX)
	save_output(target, run_num, first_passage, mean_occupancy_mot, mean_occupancy_flanks,
					mean_occupancy_local)


# sensitivity analysis for DNA concentration
def dna_concentration_sensitivity(target, run_num, y0, factors):
	DNA_concentrations = get_run_vars(target)
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(
		len(factors), len(DNA_concentrations))

	for i, factor in enumerate(factors):
		for j, DNA_concentration in enumerate(DNA_concentrations):
			k_array = get_k_array(y0[-1], factor)
			sim_data, first_passage_time = simulate_tf_search(SIM_TIME, MAX_TIME, y0, k_array, DNA=DNA_concentration)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, FLANK_INDEX)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX,
																FLANK_INDEX)
	save_output(target, run_num, first_passage, mean_occupancy_mot, mean_occupancy_flanks,
					mean_occupancy_local)


# sensitivity analysis for local volume
def local_volume_sensitivity(target, run_num, y0, factors):
	local_volumes = get_run_vars(target)
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(
		len(factors), len(local_volumes))

	for i, factor in enumerate(factors):
		for j, local_vol in enumerate(local_volumes):
			k_array = get_k_array(y0[-1], factor, local_vol=local_vol)
			sim_data, first_passage_time = simulate_tf_search(SIM_TIME, MAX_TIME, y0, k_array)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, FLANK_INDEX)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX,
																FLANK_INDEX)
	save_output(target, run_num, first_passage, mean_occupancy_mot, mean_occupancy_flanks,
					mean_occupancy_local)


# test for the occupancy function to make sure it is working correctly
def test_occupancy_function():
	sim_data = np.asarray(
		[[0, 0, 0, 0],[4000, 1, 0, 0], [0.25, 0, 1, 0], [0.25, 1, 0, 0], [0.25, 0, 1, 0], [0.25, 0, 0, 1], [0.25, 0, 1, 0]])
	print(compute_mean_occupancy(sim_data, 2, 0))
	print(compute_mean_occupancy(sim_data, 3, 0))
	print(compute_mean_occupancy(sim_data, 2, 0, 1))
	print(compute_mean_occupancy(sim_data, 1, 0))


# runs baseline gillespie simulation
def simulation(factors, run_num, y0):
	# initialize
	first_passage = np.zeros(len(factors))
	mean_occupancy_mot = np.zeros(len(factors))
	mean_occupancy_flanks = np.zeros(len(factors))
	mean_occupancy_local = np.zeros(len(factors))

	# loop through all factors and run simulation
	for j, factor in enumerate(factors):
		k_array = get_k_array(y0[-1], factor)
		sim_data, first_passage_time = simulate_tf_search(SIM_TIME, MAX_TIME, y0, k_array)
		first_passage[j] = first_passage_time
		print('factor: ', factor, ' first passage: ', first_passage_time)
		mean_occupancy_mot[j] = compute_mean_occupancy(sim_data, MOTIF_INDEX)
		mean_occupancy_flanks[j] = compute_mean_occupancy(sim_data, FLANK_INDEX)
		mean_occupancy_local[j] = compute_mean_occupancy(sim_data, MOTIF_INDEX, FLANK_INDEX)

		np.save('simulation_output/first_passage_' + run_num + '.npy', first_passage)
		np.save('simulation_output/mean_occupancy_mot_' + run_num + '.npy', mean_occupancy_mot)
		np.save('simulation_output/mean_occupancy_flanks_' + run_num + '.npy', mean_occupancy_flanks)
		np.save('simulation_output/mean_occupancy_local_' + run_num + '.npy', mean_occupancy_local)


# runs one simulation, prints information and plots local, flank and motif bound tfs over time
def one_simulation(y0):
	k_array_rpt = get_k_array(y0[-1], REPEAT_FACTOR)
	print('k_array_rpt: ', k_array_rpt)
	sim_data_rpt, first_passage = simulate_tf_search(SIM_TIME, MAX_TIME, y0, k_array_rpt)
	print('rpt first passage: ', first_passage)

	k_array_rand = get_k_array(y0[-1], RANDOM_FACTOR)
	print('k_array_rand: ', k_array_rand)
	sim_data_rand, first_passage = simulate_tf_search(SIM_TIME, MAX_TIME, y0, k_array_rand)
	print('rand first passage: ', first_passage)
	print_occupancy_info(sim_data_rand, sim_data_rpt)

	plot_time = 1000
	plot_single_molecule_trace(sim_data_rpt[:, TIME_INDEX], sim_data_rpt[:, MOTIF_INDEX],
							   sim_data_rand[:, TIME_INDEX], sim_data_rand[:, MOTIF_INDEX],
							   plot_time, 'bound to motif', 'simulation_tfs_motif')
	plot_single_molecule_trace(sim_data_rpt[:, TIME_INDEX], sim_data_rpt[:, FLANK_INDEX],
							   sim_data_rand[:, TIME_INDEX], sim_data_rand[:, FLANK_INDEX],
							   plot_time, 'bound to flanks', 'simulation_tfs_flanks')
	plot_single_molecule_trace(sim_data_rpt[:, TIME_INDEX], sim_data_rpt[:, LOCAL_INDEX],
							   sim_data_rand[:, TIME_INDEX], sim_data_rand[:, LOCAL_INDEX],
							   plot_time, 'in local volume', 'simulation_tfs_local')

	plot_trace_all_states(sim_data_rpt, sim_data_rand, plot_time)


# runs simulation of only mfpt
def mfpt_simulation(run_num, y0):
	# initialization
	run_num = int(run_num)
	rpt_fpt = np.zeros(run_num)
	rand_fpt = np.zeros(run_num)
	rpt_mfpt_mode = np.zeros(2)
	rand_mfpt_mode = np.zeros(2)

	for i in tqdm.tqdm(range(int(run_num))):
		k_array_rpt = get_k_array(y0[-1], REPEAT_FACTOR)
		sim_data_rpt, first_passage = simulate_tf_search(1e5, 1e6, y0, k_array_rpt, mfpt_only=True)
		rpt_fpt[i] = first_passage

		k_array_rand = get_k_array(y0[-1], RANDOM_FACTOR)
		sim_data_rand, first_passage = simulate_tf_search(1e5, 1e6, y0, k_array_rand, mfpt_only=True)
		rand_fpt[i] = first_passage

		# keeps track of mode for first passage (L-->F-->M or L-->M)
		if sim_data_rpt[:, -1][-1] == 2:
			rpt_mfpt_mode[1] += 1
		else:
			rpt_mfpt_mode[0] += 1
		if sim_data_rand[:, -1][-1] == 2:
			rand_mfpt_mode[1] += 1
		else:
			rand_mfpt_mode[0] += 1

	# prints summary statistics
	print('rpt mfpt: ', np.round(np.average(rpt_fpt)), ', stdev: ', np.round(np.std(rpt_fpt)),
		  ', mode: ', rpt_mfpt_mode/np.sum(rpt_mfpt_mode))
	print('rand mfpt: ', np.round(np.average(rand_fpt)), ', stdev: ', np.round(np.std(rand_fpt)),
		  ', mode: ', rand_mfpt_mode/np.sum(rand_mfpt_mode))

	# save data
	np.save('mfpt/rpt_fpt.npy', rpt_fpt)
	np.save('mfpt/rand_fpt.npy', rand_fpt)

	plot_mfpt(rand_fpt, rpt_fpt)


### HELPER FUNCTIONS


# computes the fraction of time that the target (or optionally two targets) is occupied
def compute_fraction_time_occupied(simulation_data, target_idx, target_idx_2=None):
    target_data = simulation_data[:-1, target_idx]
    if target_idx_2 is not None:
        target_data = np.add(target_data, simulation_data[:-1, target_idx_2])
    tot_occupied_time = np.sum(simulation_data[(np.where(target_data > 0)[0] + 1), DT_INDEX])
    return tot_occupied_time/simulation_data[-1, TIME_INDEX]


# computes the average occupancy of the target (or optionally two targets)
def compute_mean_occupancy(simulation_data, target_idx, target_idx_2=None):
	if target_idx_2 is not None:
		return np.average(np.add(simulation_data[:-1, target_idx],
								 simulation_data[:-1, target_idx_2]),
						  weights=simulation_data[1:, DT_INDEX])
	return np.average(simulation_data[:-1, target_idx], weights=simulation_data[1:, DT_INDEX])


# saves simulation results to .npy files
def save_output(target, run_num, first_passage, mean_occupancy_mot,
				mean_occupancy_flanks, mean_occupancy_local):
    np.save('simulation_output/' + target + '_sensitivity_first_passage_' + run_num + '.npy', first_passage)
    np.save('simulation_output/' + target + '_sensitivity_mean_occupancy_mot_' + run_num + '.npy', mean_occupancy_mot)
    np.save('simulation_output/' + target + '_sensitivity_mean_occupancy_flanks_' + run_num + '.npy', mean_occupancy_flanks)
    np.save('simulation_output/' + target + '_sensitivity_mean_occupancy_local_' + run_num + '.npy', mean_occupancy_local)


# creates storage variables for simulation results
def initialize_storage(n_factors, n_var):
    return np.zeros((n_factors, n_var)), np.zeros((n_factors, n_var)), np.zeros((n_factors, n_var)), np.zeros((n_factors, n_var))


# returns variables for sensitivity analyses
def get_run_vars(target):
	if target == 'n_flanks':
		return np.geomspace(1, 1000, 10)
	if target == 'core_affinity':
		return np.geomspace(1e-9, 1e-5, 10)
	if target == 'koff_slope':
		return np.geomspace(0.1, 1, 10)
	if target == 'koff_intercept':
		return np.geomspace(-3, 5, 10)
	if target == 'n_TF':
		return np.geomspace(1, 1000, 10)
	if target == 'switching_rate':
		return np.geomspace(0.01, 1e2, 10)
	if target == 'diffusion':
		return np.geomspace(0.5, 5, 10)
	if target == 'DNA_concentration':
		return np.geomspace(1e-7, 1e-3, 10)
	if target == 'local_volume':
		return np.geomspace(0.5e-4, 5e-4, 10)


# prints occupancy information about one simulation
def print_occupancy_info(sim_data_rand, sim_data_rpt):
	print(
		'                               [cellular milleu, local, motif bound, flanks bound]')
	print('')
	print('repeat fraction time occupied: ',
		  [compute_fraction_time_occupied(sim_data_rpt, x, 0, 1) for x in
		   list(np.arange(2, 6))])
	print('random fraction time occupied: ',
		  [compute_fraction_time_occupied(sim_data_rand, x, 0, 1) for x in
		   list(np.arange(2, 6))])
	print('repeat mean occupancy: ',
		  [compute_mean_occupancy(sim_data_rpt, x, 0) for x in np.arange(2, 6)])
	print('random mean occupancy: ',
		  [compute_mean_occupancy(sim_data_rand, x, 0) for x in np.arange(2, 6)])
	print('')
	print('repeat mean number of TFs bound in antennae: ',
		  compute_mean_occupancy(sim_data_rpt, 4, 0, 5))
	print('random mean number of TFs bound in antennae: ',
		  compute_mean_occupancy(sim_data_rand, 4, 0, 5))
	print('')
	print('repeat fraction of time with TF bound in antennae: ',
		  compute_fraction_time_occupied(sim_data_rpt, 4, 0, 1, 5))
	print('random fraction of time with TF bound in antennae: ',
		  compute_fraction_time_occupied(sim_data_rand, 4, 0, 1, 5))


# plots example single molecule trace of gillespie algorithm for one state
def plot_single_molecule_trace(x1, y1, x2, y2, plot_time, location, title):
	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax1.plot(x1, y1, alpha=0.5, drawstyle='steps-post')
	ax1.set_xlim([0, plot_time])
	ax1.set_title('TFs ' + location + ' with repeat flanks')
	ax1.set_ylabel('# of molecules')
	ax1.set_yticks([0, 1])
	ax1.set_xlabel('time (s)')
	ax2 = fig.add_subplot(212)
	ax2.plot(x2, y2, alpha=0.5, drawstyle='steps-post')
	ax2.set_title('TFs ' + location + ' to motif with random flanks')
	ax2.set_xlim([0, plot_time])
	ax2.set_yticks([0, 1])
	ax2.set_ylabel('# of molecules')
	ax2.set_xlabel('time (s)')
	plt.tight_layout()
	plt.savefig(title + '.pdf')


# plots example single molecule trace of all four states together
def plot_trace_all_states(sim_data_rpt, sim_data_rand, plot_time):
	# plot of all four states
	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	state_data = sim_data_rpt[:,
				 NUCLEUS_INDEX] + sim_data_rpt[:,
								  LOCAL_INDEX]*2 + sim_data_rpt[:,
												   FLANK_INDEX]*3 + sim_data_rpt[:,
																	MOTIF_INDEX]*4
	ax1.plot(sim_data_rpt[:, TIME_INDEX], state_data, alpha=0.5, drawstyle='steps-post')
	ax1.set_xlim([0, plot_time])
	ax1.set_title('TFs in local volume with repeat flanks')
	ax1.set_ylabel('# of molecules')
	ax1.set_yticks([1, 2, 3, 4])
	ax1.set_yticklabels(['CM', 'local', 'flanks', 'motif'])
	ax1.set_xlabel('time (s)')
	ax2 = fig.add_subplot(212)
	state_data = sim_data_rand[:,
				 NUCLEUS_INDEX] + sim_data_rand[:,
								  LOCAL_INDEX] * 2 + sim_data_rand[:,
													 FLANK_INDEX] * 3 + sim_data_rand[:,
																		MOTIF_INDEX] * 4
	ax2.plot(sim_data_rand[:, TIME_INDEX], state_data, alpha=0.5, drawstyle='steps-post')
	ax2.set_xlim([0, plot_time])
	ax2.set_title('TFs in local volume with random flanks')
	ax2.set_ylabel('# of molecules')
	ax2.set_yticks([1, 2, 3, 4])
	ax2.set_yticklabels(['CM', 'local', 'flanks', 'motif'])
	ax2.set_xlabel('time (s)')
	plt.tight_layout()
	plt.savefig('simulation_summary.pdf')


# plots MFPT for repeat and random with SEM
def plot_mfpt(rand_fpt, rpt_fpt, run_num):
	fig, ax = plt.subplots()
	ax.bar([0, 1], [np.average(rand_fpt), np.average(rpt_fpt)], yerr=[np.std(rand_fpt)/np.sqrt(run_num), np.std(rpt_fpt)/np.sqrt(run_num)],
		   align='center', alpha=0.5, ecolor='black', capsize=10)
	# ax.violinplot([rand_fpt, rpt_fpt], [0, 1], showmeans=True, showextrema=True)
	ax.set_ylabel('Mean first passage time (s)')
	ax.set_xticks([0, 1])
	ax.set_xticklabels(['random', 'repeat'])
	ax.set_title('MFPT for random and repeat flanks (N = %d)'%run_num)
	ax.yaxis.grid(True)

	# Save the figure and show
	plt.tight_layout()
	plt.savefig('mfpt/MFPT_plot.pdf')


if __name__ == "__main__":
	main()
