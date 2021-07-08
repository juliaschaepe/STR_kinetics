import numpy as np
import argparse
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Global variables
MOTIF_INDEX = 4
FLANK_INDEX = 5
DT_INDEX = 0
TIME_INDEX = 1
SIM_TIME = 5e3
MAX_TIME = 1e5

# returns kinetic parameters for gillespie simulation
def get_k_array(n_flanks, factor, core_affinity=1e-7,
				sliding_kinetics = 0.5, tf_diffusion=1, koff_slope=0.2031033040149204):
	koff_intercept = 0.41372462865960724

	nuc_vol = 3 # um^3
	local_vol = 1e-4 # um^3
	D = tf_diffusion  # um^2/s
	R = np.cbrt(3 * local_vol / (4 * np.pi)) # um

	Kd_23 = core_affinity  # source: MITOMI data
	Kd_24 = core_affinity / factor
	Kd_34 = Kd_24 / Kd_23
	Kd_12 = nuc_vol / local_vol

	k12 = 4 * np.pi * D * R / nuc_vol # 1/s
	k21 = k12 * Kd_12 # to maintain "detailed balance", 1/s

	k32 = np.exp(np.log(Kd_23)*koff_slope + koff_intercept) # source: MITOMI data - this is koff, 1/s
	k23 = k32 / Kd_23  # detailed balance - this is kon, 1/Ms

	k42 = np.exp(np.log(Kd_24)*koff_slope + koff_intercept)  # source: MITOMI data - this is koff, 1/s
	k24 = k42 / Kd_24  # detailed balance - this is kon, 1/Ms

	# k43 = sliding_kinetics / 62.5 / 7 # this might need to be re-evaluated, 1/s
	# k43 = 100 / 62.5 / 7
	# k43 = 0.5
	k43 = sliding_kinetics
	k34 = k43 / Kd_34  # detailed balance, 1/s

	return k12, k21, k23, k24, k32, k34, k42, k43


# Gillespie simulation of TFsearch
def simulate_tf_search(sim_T, max_T, y0, k_array, DNA=5e-5, mfpt_only = False):
	stored_data = False
	first_passage = False
	first_flanks = False
	first_flanks_time = -1
	first_passage_time = -1
	rxn = -1
	t = 0
	k12, k21, k23, k24, k32, k34, k42, k43 = k_array
	i = 1
	y = y0.copy()
	n_rows = 100000
	sim_data = np.zeros((n_rows, len(y0) + 3))
	sim_data[0] = np.hstack([0, t, y, rxn])
	w_mapping = [([0], [1]),
				 ([1], [0]), ([1, 4], [2]), ([1, 5], [3]),
				 ([2], [1, 4]), ([2, 5], [3, 4]),
				 ([3], [1, 5]), ([3, 4], [2, 5])]

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

		# ensures conservation of 1 motif and 100 flanks without altering rates
		if y[4] == 0:
			w43 = 0
		if y[5] == 0:
			w34 = 0

		# calculates tau
		w_arr = [w12, w21, w23, w24, w32, w34, w42, w43]
		# print(w_arr)
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
		# print(t)
		if y[5] != y0[-1] and not first_flanks:
			first_flanks_time = t
			first_flanks = True
		if y[4] == 0 and not first_passage:
			first_passage_time = t
			first_passage = True
		# print('first_passage: ', t)

		# allocates more space so that sim_data is not stored dynamically
		if i >= n_rows:
			sim_data = np.vstack((sim_data, np.zeros((n_rows, len(y0) + 3))))
			n_rows += n_rows

		# updates sim_data
		sim_data[i] = np.hstack([tau, t, y, rxn])
		i += 1

		if t >= sim_T and not stored_data:
			sim_data_occ = np.asarray(sim_data)
			sim_data_occ = sim_data_occ[:np.argmax(sim_data_occ[:, 1]) + 1, :]
			stored_data = True
		if mfpt_only and first_passage:
			sim_data_occ = np.asarray(sim_data)
			sim_data_occ = sim_data_occ[:np.argmax(sim_data_occ[:, 1]) + 1, :]
			return sim_data_occ, first_passage_time, first_flanks_time
		if stored_data and first_passage:
			return sim_data_occ, first_passage_time
	return sim_data_occ, max_T*10

# computes the fraction of time that the target is occupied
def compute_fraction_time_occupied(simulation_data, target_idx, dt_index, time_index, target_idx_2 = None):
    target_data = simulation_data[:-1, target_idx]
    if target_idx_2 is not None:
        target_data = np.add(target_data, simulation_data[:-1, target_idx_2])
    tot_occupied_time = np.sum(simulation_data[(np.where(target_data > 0)[0] + 1), dt_index])
    return tot_occupied_time/simulation_data[-1,time_index]


# computes the average occupancy of the target
def compute_mean_occupancy(simulation_data, target_idx, dt_index, target_idx_2 = None):
	if target_idx_2 is not None:
		return np.average(np.add(simulation_data[:-1, target_idx], simulation_data[:-1, target_idx_2]), weights=simulation_data[1:, dt_index])
	return np.average(simulation_data[:-1, target_idx], weights=simulation_data[1:, dt_index])


# saves simulation results to .npy files
def save_output(target, run_num, first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local):
    np.save('simulation_output/' + target + '_sensitivity_first_passage_' + run_num + '.npy', first_passage)
    np.save('simulation_output/' + target + '_sensitivity_mean_occupancy_mot_' + run_num + '.npy', mean_occupancy_mot)
    np.save('simulation_output/' + target + '_sensitivity_mean_occupancy_flanks_' + run_num + '.npy', mean_occupancy_flanks)
    np.save('simulation_output/' + target + '_sensitivity_mean_occupancy_local_' + run_num + '.npy', mean_occupancy_local)


# creates storage variables for simulation results
def initialize_storage(n_factor, n_var):
    return np.zeros((n_factor, n_var)), np.zeros((n_factor, n_var)), np.zeros((n_factor, n_var)), np.zeros((n_factor, n_var))


# sensitivity analysis for number of flanks
def n_flanks_sensitivity(target, run_num, y0, factor):
	n_flanks = np.array([1, 10, 100, 500, 1000])
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(len(factor), len(n_flanks))
	for i, ratio in enumerate(factor):
		for j, n in enumerate(n_flanks):
			y0[-1] = n
			k_array = get_k_array(n, ratio)
			sim_data, first_passage_time = simulate_tf_search(SIM_TIME, MAX_TIME, y0, k_array)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX, DT_INDEX)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, FLANK_INDEX,
																 DT_INDEX)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX,
																DT_INDEX, FLANK_INDEX)
	save_output(target, run_num, first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local)


# sensitivity analysis for motif affinity
def core_affinity_sensitivity(target, run_num, y0, factor):
	core_affinity = np.geomspace(1e-10, 1e-6, 10)
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(len(factor), len(core_affinity))
	for i, ratio in enumerate(factor):
		for j, Kd in enumerate(core_affinity):
			k_array = get_k_array(100, ratio, core_affinity=Kd)
			sim_data, first_passage_time = simulate_tf_search(SIM_TIME, MAX_TIME, y0, k_array)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX, DT_INDEX)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, FLANK_INDEX,
																 DT_INDEX)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX,
																DT_INDEX, FLANK_INDEX)
	save_output(target, run_num, first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local)


# sensitivity analysis for koff kinetics
def koff_sensitivity(target, run_num, y0, factor):
	koff_slope = np.geomspace(0.1, 1, 10)
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(len(factor), len(koff_slope))
	for i, ratio in enumerate(factor):
		for j, slope in enumerate(koff_slope):
			k_array = get_k_array(100, ratio, koff_slope=slope)
			sim_data, first_passage_time = simulate_tf_search(SIM_TIME, MAX_TIME, y0, k_array)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX, DT_INDEX)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, FLANK_INDEX,
																 DT_INDEX)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX,
																DT_INDEX, FLANK_INDEX)
	save_output(target, run_num, first_passage, mean_occupancy_mot, mean_occupancy_flanks,
				mean_occupancy_local)


# sensitivity analysis for number of TFs
def n_tf_sensitivity(target, run_num, y0, factor):
	TF_number = np.array([1, 5, 10, 50, 100])
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(len(factor), len(TF_number))
	for i, ratio in enumerate(factor):
		for j, n_TF in enumerate(TF_number):
			k_array = get_k_array(100, ratio)
			y0[0] = n_TF
			sim_data, first_passage_time = simulate_tf_search(SIM_TIME, MAX_TIME, y0, k_array)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX, DT_INDEX)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, FLANK_INDEX,
																 DT_INDEX)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX,
																DT_INDEX, FLANK_INDEX)
	save_output(target, run_num, first_passage, mean_occupancy_mot, mean_occupancy_flanks,
					mean_occupancy_local)


# sensitivity analysis for 1D TF-DNA sliding rate
def sliding_kinetics_sensitivity(target, run_num, y0, factor):
	sliding_kinetics = np.geomspace(0.1, 1e4, 10)
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(len(factor), len(sliding_kinetics))

	for i, ratio in enumerate(factor):
		print(ratio)
		for j, value in enumerate(sliding_kinetics):
			k_array = get_k_array(100, ratio, sliding_kinetics=value)
			sim_data, first_passage_time = simulate_tf_search(SIM_TIME, MAX_TIME, y0, k_array)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX, DT_INDEX)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, FLANK_INDEX,
																 DT_INDEX)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX,
																DT_INDEX, FLANK_INDEX)
	save_output(target, run_num, first_passage, mean_occupancy_mot, mean_occupancy_flanks,
					mean_occupancy_local)


# sensitivity analysis for rate of diffusion
def diffusion_kinetics_sensitivity(target, run_num, y0, factor):
	# TODO: this needs to be updated
	diff_3D_arr = np.geomspace(1e-3, 1e2, 10)
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(len(factor), len(diff_3D_arr))
	for i, ratio in enumerate(factor):
		for j, d in enumerate(diff_3D_arr):
			k_array = get_k_array(100, ratio, tf_diffusion=d)
			sim_data, first_passage_time = simulate_tf_search(SIM_TIME, MAX_TIME, y0, k_array)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX, DT_INDEX)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, FLANK_INDEX,
																 DT_INDEX)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX,
																DT_INDEX, FLANK_INDEX)
	save_output(target, run_num, first_passage, mean_occupancy_mot, mean_occupancy_flanks,
					mean_occupancy_local)


# sensitivity analysis for DNA concentration
def dna_concentration_sensitivity(target, run_num, y0, factor):
	DNA_conc_arr = np.geomspace(1e-7, 1e-3, 10)
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(len(factor), len(DNA_conc_arr))

	for i, ratio in enumerate(factor):
		for j, DNA_conc in enumerate(DNA_conc_arr):
			k_array = get_k_array(100, ratio)
			sim_data, first_passage_time = simulate_tf_search(SIM_TIME, MAX_TIME, y0, k_array, DNA=DNA_conc)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX, DT_INDEX)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, FLANK_INDEX,
																 DT_INDEX)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, MOTIF_INDEX,
																DT_INDEX, FLANK_INDEX)
	save_output(target, run_num, first_passage, mean_occupancy_mot, mean_occupancy_flanks,
					mean_occupancy_local)


# runs baseline gillespie simulation
def simulation(target, run_num, y0):
	factor = np.geomspace(1e-4, 10)
	n_deg_sites = 100
	first_passage = np.zeros(len(factor))
	mean_occupancy_mot = np.zeros(len(factor))
	mean_occupancy_flanks = np.zeros(len(factor))
	mean_occupancy_local = np.zeros(len(factor))
	for j, ratio in enumerate(factor):
		k_array = get_k_array(n_deg_sites, ratio)
		sim_data, first_passage_time = simulate_tf_search(SIM_TIME, MAX_TIME, y0, k_array)
		first_passage[j] = first_passage_time
		print('ratio: ', ratio, ' first passage: ', first_passage_time)
		mean_occupancy_mot[j] = compute_mean_occupancy(sim_data, MOTIF_INDEX, DT_INDEX)
		mean_occupancy_flanks[j] = compute_mean_occupancy(sim_data, FLANK_INDEX, DT_INDEX)
		mean_occupancy_local[j] = compute_mean_occupancy(sim_data, MOTIF_INDEX, DT_INDEX, FLANK_INDEX)

		np.save('simulation_output/first_passage_' + run_num + '.npy', first_passage)
		np.save('simulation_output/mean_occupancy_mot_' + run_num + '.npy', mean_occupancy_mot)
		np.save('simulation_output/mean_occupancy_flanks_' + run_num + '.npy', mean_occupancy_flanks)
		np.save('simulation_output/mean_occupancy_local_' + run_num + '.npy', mean_occupancy_local)


# tests the occupancy function to make sure it is working correctly
def test_occupancy_function():
	sim_data = np.asarray(
		[[0, 0, 0, 0],[4000, 1, 0, 0], [0.25, 0, 1, 0], [0.25, 1, 0, 0], [0.25, 0, 1, 0], [0.25, 0, 0, 1], [0.25, 0, 1, 0]])
	print(compute_mean_occupancy(sim_data, 2, 0))
	print(compute_mean_occupancy(sim_data, 3, 0))
	print(compute_mean_occupancy(sim_data, 2, 0, 1))
	print(compute_mean_occupancy(sim_data, 1, 0))


# runs one simulation and plots local, flank and motif bound tfs over time
def one_simulation(y0):
	k_array_rpt = get_k_array(100, 0.1)
	sim_time = 5e3
	print('k_array_rpt: ', k_array_rpt)
	sim_data_rpt, first_passage = simulate_tf_search(sim_time, 1e4, y0, k_array_rpt)
	print('rpt first passage: ', first_passage)

	k_array_rand = get_k_array(100, 0.01)
	print('k_array_rand: ', k_array_rand)
	sim_data_rand, first_passage = simulate_tf_search(sim_time, 1e4, y0, k_array_rand)
	print('rand first passage: ', first_passage)
	print(
		'                               [cellular milleu, local, motif bound, flanks bound]')
	print('')
	print('repeat fraction time occupied: ',
		  [compute_fraction_time_occupied(sim_data_rpt, x, 0, 1) for x in list(np.arange(2, 6))])
	print('random fraction time occupied: ',
		  [compute_fraction_time_occupied(sim_data_rand, x, 0, 1) for x in list(np.arange(2, 6))])
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
	sim_time = 1000
	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax1.plot(sim_data_rpt[:, 1], sim_data_rpt[:, 4], alpha=0.5, drawstyle='steps-post')
	ax1.set_xlim([0, sim_time])
	ax1.set_title('TFs bound to motif with repeat flanks')
	ax1.set_ylabel('# of molecules')
	ax1.set_yticks([0,1])
	ax1.set_xlabel('time (s)')
	ax2 = fig.add_subplot(212)
	ax2.plot(sim_data_rand[:, 1], sim_data_rand[:, 4], alpha=0.5, drawstyle='steps-post')
	ax2.set_title('TFs bound to motif with random flanks')
	ax2.set_xlim([0, sim_time])
	ax2.set_yticks([0,1])
	ax2.set_ylabel('# of molecules')
	ax2.set_xlabel('time (s)')
	plt.tight_layout()
	plt.savefig('simulation_tfs_motif.pdf')

	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax1.plot(sim_data_rpt[:, 1], sim_data_rpt[:, 5], alpha=0.5, drawstyle='steps-post')
	ax1.set_xlim([0, sim_time])
	ax1.set_title('TFs bound to flanks with repeat flanks')
	ax1.set_ylabel('# of molecules')
	ax1.set_yticks([0, 1])
	ax1.set_xlabel('time (s)')
	ax2 = fig.add_subplot(212)
	ax2.plot(sim_data_rand[:, 1], sim_data_rand[:, 5], alpha=0.5, drawstyle='steps-post')
	ax2.set_title('TFs bound to flanks with random flanks')
	ax2.set_xlim([0, sim_time])
	ax2.set_yticks([0, 1])
	ax2.set_ylabel('# of molecules')
	ax2.set_xlabel('time (s)')
	plt.tight_layout()
	plt.savefig('simulation_tfs_flanks.pdf')

	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax1.plot(sim_data_rpt[:, 1], sim_data_rpt[:, 3], alpha=0.5, drawstyle='steps-post')
	ax1.set_xlim([0, sim_time])
	ax1.set_title('TFs in local volume with repeat flanks')
	ax1.set_ylabel('# of molecules')
	ax1.set_yticks([0, 1])
	ax1.set_xlabel('time (s)')
	ax2 = fig.add_subplot(212)
	ax2.plot(sim_data_rand[:, 1], sim_data_rand[:, 3], alpha=0.5, drawstyle='steps-post')
	ax2.set_title('TFs in local volume with random flanks')
	ax2.set_xlim([0, sim_time])
	ax2.set_yticks([0, 1])
	ax2.set_ylabel('# of molecules')
	ax2.set_xlabel('time (s)')
	plt.tight_layout()
	plt.savefig('simulation_tfs_local.pdf')

	# plot of all four states
	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	state_data = sim_data_rpt[:, 2] + sim_data_rpt[:, 3]*2 + sim_data_rpt[:, 5]*3 + sim_data_rpt[:, 4]*4
	ax1.plot(sim_data_rpt[:, 1], state_data, alpha=0.5, drawstyle='steps-post')
	# for i in range(len(sim_data_rpt)):
	# 	if sim_data_rpt[i, 1] >= sim_time:
	# 		break
	# 	if sim_data_rpt[i, 4] == 1:
	# 		ax1.axvspan(sim_data_rpt[i, 1], sim_data_rpt[i+1, 1], color='orange', alpha=0.5, capstyle='butt')
	ax1.set_xlim([0, sim_time])
	ax1.set_title('TFs in local volume with repeat flanks')
	ax1.set_ylabel('# of molecules')
	ax1.set_yticks([1, 2, 3, 4])
	ax1.set_yticklabels(['CM', 'local', 'flanks', 'motif'])
	ax1.set_xlabel('time (s)')
	ax2 = fig.add_subplot(212)
	state_data = sim_data_rand[:, 2] + sim_data_rand[:, 3] * 2 + sim_data_rand[:,
															   5] * 3 + sim_data_rand[:, 4] * 4
	ax2.plot(sim_data_rand[:, 1], state_data, alpha=0.5, drawstyle='steps-post')
	# for i in range(len(sim_data_rand)):
	# 	if sim_data_rand[i, 1] >= sim_time:
	# 		break
	# 	if sim_data_rand[i, 4] == 1:
	# 		ax2.axvspan(sim_data_rand[i, 1], sim_data_rand[i+1, 1], color='orange', alpha=0.5)
	ax2.set_xlim([0, sim_time])
	ax2.set_title('TFs in local volume with random flanks')
	ax2.set_ylabel('# of molecules')
	ax2.set_yticks([1, 2, 3, 4])
	ax2.set_yticklabels(['CM', 'local', 'flanks', 'motif'])
	ax2.set_xlabel('time (s)')
	plt.tight_layout()
	plt.savefig('simulation_summary.pdf')


def mfpt_simulation(run_num, y0):
	run_num = int(run_num)
	rpt_fpt = np.zeros(run_num)
	rand_fpt = np.zeros(run_num)
	rpt_flanks = np.zeros(run_num)
	rand_flanks = np.zeros(run_num)
	rpt_mfpt_mode = np.zeros(2)
	rand_mfpt_mode = np.zeros(2)
	for i in tqdm.tqdm(range(int(run_num))):
		k_array_rpt = get_k_array(100, 0.1)
		sim_data_rpt, first_passage, first_flanks = simulate_tf_search(1e5, 1e6, y0, k_array_rpt, mfpt_only=True)
		rpt_fpt[i] = first_passage
		rpt_flanks[i] = first_flanks

		k_array_rand = get_k_array(100, 0.01)
		sim_data_rand, first_passage, first_flanks = simulate_tf_search(1e5, 1e6, y0, k_array_rand, mfpt_only=True)
		rand_fpt[i] = first_passage
		rand_flanks[i] = first_flanks

		if sim_data_rpt[:, -1][-1] == 2:
			rpt_mfpt_mode[1] += 1
		else:
			rpt_mfpt_mode[0] += 1
		if sim_data_rand[:, -1][-1] == 2:
			rand_mfpt_mode[1] += 1
		else:
			rand_mfpt_mode[0] += 1
		# print('rpt rxns: ', sim_data_rpt[:, -1])
		# print('rpt taus: ', sim_data_rpt[:, 0])
		# print('rand rxns: ', sim_data_rand[:, -1])
		# print('rand taus: ', sim_data_rand[:, 0])

	print('rpt mfpt: ', np.round(np.average(rpt_fpt)), ', stdev: ', np.round(np.std(rpt_fpt)),
		  ', mode: ', rpt_mfpt_mode/np.sum(rpt_mfpt_mode))
	print('rand mfpt: ', np.round(np.average(rand_fpt)), ', stdev: ', np.round(np.std(rand_fpt)),
		  ', mode: ', rand_mfpt_mode/np.sum(rand_mfpt_mode))
	np.save('mfpt/rpt_fpt.npy', rpt_fpt)
	np.save('mfpt/rand_fpt.npy', rand_fpt)

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


# parses input and runs appropriate simulation or sensitivity analysis
def main():
	parser = argparse.ArgumentParser(description='Get run number and sensitivity analysis target')
	parser.add_argument('run_num', type=str, help='run number')
	parser.add_argument('target', type=str, help='sensitivity analysis target variable')
	parser.add_argument("-y0", type=int, nargs="+", default=[1, 0, 0, 0, 1, 100])
	args = parser.parse_args()
	factor = np.geomspace(1e-3, 1, num=10)
	# options for targets: 'n_flanks', 'core_affinity', 'core_kinetics', 'flank_kinetics', 'n_TF'
	# 'sliding_kinetics', 'diffusion_kinetics', 'DNA_concentration'
	if args.target == 'simulation':
		simulation(args.target, args.run_num, args.y0)
	if args.target == 'n_flanks':
		n_flanks_sensitivity(args.target, args.run_num, args.y0, factor)
	if args.target == 'core_affinity':
		core_affinity_sensitivity(args.target, args.run_num, args.y0, factor)
	if args.target == 'koff':
		koff_sensitivity(args.target, args.run_num, args.y0, factor)
	if args.target == 'n_TF':
		n_tf_sensitivity(args.target, args.run_num, args.y0, factor)
	if args.target == 'sliding_kinetics':
		sliding_kinetics_sensitivity(args.target, args.run_num, args.y0, factor)
	if args.target == 'diffusion_kinetics':
		diffusion_kinetics_sensitivity(args.target, args.run_num, args.y0, factor)
	if args.target == 'DNA_concentration':
		dna_concentration_sensitivity(args.target, args.run_num, args.y0, factor)
	if args.target == 'one_simulation':
		one_simulation(args.y0)
	if args.target == 'mfpt_simulation':
		mfpt_simulation(args.run_num, args.y0)

	print('Run completed')


if __name__ == "__main__":
	# test_occupancy_function()
	# one_simulation()
	main()
