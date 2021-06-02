import numpy as np
import argparse
import tqdm

# returns kinetic parameters for gillespie simulation
def get_k_array(n_flanks, factor, core_affinity=1e-7, core_koff=0.01, flank_koff=5,
				velocity_prob=1000 / 7, tf_diffusion=10e-7 / 100):

	nuc_vol = 3e-15
	local_vol = 3e-19
	D = tf_diffusion  # cm^2/s --> L/s
	R = np.cbrt(3 * local_vol / (4 * np.pi))

	Kd_23 = core_affinity  # source: MITOMI data
	Kd_24 = core_affinity / (factor * n_flanks)
	Kd_34 = Kd_24 / Kd_23
	Kd_12 = nuc_vol / local_vol

	k12 = 4 * np.pi * D * R / nuc_vol # 1/s
	k21 = k12 * Kd_12  # to maintain "detailed balance", 1/s

	k32 = core_koff  # source: MITOMI data - this is koff, 1/s
	k23 = k32 / Kd_23  # detailed balance - this is kon, 1/Ms

	k42 = flank_koff  # source: MITOMI data - this is koff, 1/s
	k24 = k42 / Kd_24  # detailed balance - this is kon, 1/Ms

	k43 = velocity_prob / (n_flanks / 2)  # this might need to be re-evaluated, 1/s
	k34 = k43 / Kd_34  # detailed balance, 1/s

	return k12, k21, k23, k24, k32, k34, k42, k43


# Gillespie simulation of TFsearch
def simulate_tf_search(sim_T, max_T, y0, k_array, DNA=5e-5):
	stored_data = False
	first_passage_time = -1
	t = 0
	k12, k21, k23, k24, k32, k34, k42, k43 = k_array
	i = 0
	y = y0
	n_rows = 100000
	sim_data = np.zeros((n_rows, len(y0) + 2))
	w_mapping = [([0], [1]),
				 ([1], [0]), ([1, 4], [2]), ([1, 5], [3]),
				 ([2], [1, 4]), ([2, 5], [3, 4]),
				 ([3], [1, 5]), ([3, 4], [2, 5])]

	while t < max_T:

		# calculates likelihood of each reaction
		w12 = k12 * y[0] * DNA
		w21 = k21 * y[1] * DNA
		w23 = k23 * y[1] * y[4] * DNA
		w32 = k32 * y[2]
		w24 = k24 * y[1] * y[5] * DNA
		w42 = k42 * y[3]
		w34 = k34 * y[2]
		w43 = k43 * y[3]

		# ensures conservation of 1 motif and 100 flanks without altering with rates
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
			if rand <= (np.sum(w_arr[:j + 1]) / W):
				if j == 3:
					first_passage_time = t
				idx_from, idx_to = w_mapping[j]
				for idx in idx_from:
					y[idx] -= 1
				for idx in idx_to:
					y[idx] += 1
				break

		# allocates more space so that sim_data is not stored dynamically
		if i >= n_rows:
			sim_data = np.vstack((sim_data, np.zeros((n_rows, len(y0) + 2))))
			n_rows += n_rows

		# updates sim_data
		sim_data[i] = np.hstack([tau, t, y])
		i += 1

		if t >= sim_T:
			sim_data_occ = np.asarray(sim_data)
			sim_data_occ = sim_data_occ[:np.argmax(sim_data_occ[:, 1]) + 1, :]
			stored_data = True
		if stored_data and first_passage_time != -1:
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
			sim_data, first_passage_time = simulate_tf_search(1e3, 1e5, y0, k_array)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, 3, 0)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, 4, 0)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, 3, 0, 4)
	save_output(target, run_num, first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local)


# sensitivity analysis for motif affinity
def core_affinity_sensitivity(target, run_num, y0, factor):
	core_affinity = np.geomspace(1e-10, 1e-6, 10)
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(len(factor), len(core_affinity))
	for i, ratio in enumerate(factor):
		for j, Kd in enumerate(core_affinity):
			k_array = get_k_array(100, ratio, core_affinity=Kd)
			sim_data, first_passage_time = simulate_tf_search(1e3, 1e5, y0, k_array)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, 3, 0)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, 4, 0)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, 3, 0, 4)
	save_output(target, run_num, first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local)


# sensitivity analysis for motif kinetics
def core_kinetics_sensitivity(target, run_num, y0, factor):
	core_koff = np.geomspace(1e-3, 1, 10)
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(len(factor), len(core_koff))
	for i, ratio in enumerate(factor):
		for j, koff in enumerate(core_koff):
			k_array = get_k_array(100, ratio, core_koff=koff)
			sim_data, first_passage_time = simulate_tf_search(1e3, 1e5, y0, k_array)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, 3, 0)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, 4, 0)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, 3, 0, 4)
	save_output(target, run_num, first_passage, mean_occupancy_mot, mean_occupancy_flanks,
				mean_occupancy_local)


# sensitivity analysis for flank kinetics
def flank_kinetics_sensitivity(target, run_num, y0, factor):
	koff_factors = np.geomspace(1e-3, 10, 10)
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(len(factor), len(koff_factors))
	for i, ratio in enumerate(factor):
		for j, koff_factor in enumerate(koff_factors):
			k_array = get_k_array(100, ratio, koff_factor = koff_factor)
			sim_data, first_passage_time = simulate_tf_search(1e3, 1e5, y0, k_array)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, 3, 0)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, 4, 0)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, 3, 0, 4)
	save_output(target, run_num, first_passage, mean_occupancy_mot, mean_occupancy_flanks,
					mean_occupancy_local)


# sensitivity analysis for number of TFs
def n_tf_sensitivity(target, run_num, y0, factor):
	TF_number = np.array([50, 100, 200, 500, 1000, 2000, 5000])
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(len(factor), len(TF_number))
	for i, ratio in enumerate(factor):
		for j, n_TF in enumerate(TF_number):
			k_array = get_k_array(100, ratio)
			y0[0] = n_TF
			sim_data, first_passage_time = simulate_tf_search(1e3, 1e5, y0, k_array)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, 3, 0)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, 4, 0)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, 3, 0, 4)
	save_output(target, run_num, first_passage, mean_occupancy_mot, mean_occupancy_flanks,
					mean_occupancy_local)


# sensitivity analysis for 1D TF-DNA sliding rate
def sliding_kinetics_sensitivity(target, run_num, y0, factor):
	velocity_x_probability = np.geomspace(5, 1000, 10)
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(len(factor), len(velocity_x_probability))

	for i, ratio in enumerate(factor):
		for j, vxpb in enumerate(velocity_x_probability):
			k_array = get_k_array(100, ratio, velocity_prob=vxpb)
			sim_data, first_passage_time = simulate_tf_search(1e3, 1e5, y0, k_array)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, 3, 0)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, 4, 0)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, 3, 0, 4)
	save_output(target, run_num, first_passage, mean_occupancy_mot, mean_occupancy_flanks,
					mean_occupancy_local)


# sensitivity analysis for rate of diffusion
def diffusion_kinetics_sensitivity(target, run_num, y0, factor):
	# TODO: this needs to be updated
	diff_3D_arr = np.geomspace(1e-5, 1e-2, 10)
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(len(factor), len(diff_3D_arr))
	for i, ratio in enumerate(factor):
		for j, d in enumerate(diff_3D_arr):
			k_array = get_k_array(100, ratio, tf_diffusion=d)
			sim_data, first_passage_time = simulate_tf_search(1e3, 1e5, y0, k_array)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, 3, 0)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, 4, 0)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, 3, 0, 4)
	save_output(target, run_num, first_passage, mean_occupancy_mot, mean_occupancy_flanks,
					mean_occupancy_local)


# sensitivity analysis for DNA concentration
def dna_concentration_sensitivity(target, run_num, y0, factor):
	DNA_conc_arr = np.geomspace(1e-7, 1e-3, 10)
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(len(factor), len(DNA_conc_arr))

	for i, ratio in enumerate(factor):
		for j, DNA_conc in enumerate(DNA_conc_arr):
			k_array = get_k_array(100, ratio)
			sim_data, first_passage_time = simulate_tf_search(1e3, 1e5, y0, k_array)
			first_passage[i, j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, 3, 0)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, 4, 0)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, 3, 0, 4)
	save_output(target, run_num, first_passage, mean_occupancy_mot, mean_occupancy_flanks,
					mean_occupancy_local)


# runs baseline gillespie simulation
def simulation(target, run_num, y0):
	factor = np.geomspace(1e-4, 10)
	n_deg_sites = 100
	sim_time = 1e3
	max_time = 1e5
	first_passage = np.zeros((int(run_num), len(factor)))
	mean_occupancy_mot = np.zeros((int(run_num), len(factor)))
	mean_occupancy_flanks = np.zeros((int(run_num), len(factor)))
	mean_occupancy_local = np.zeros((int(run_num), len(factor)))

	for i in tqdm.tqdm(range(int(run_num)), miniters=50):
		for j, ratio in enumerate(factor):
			k_array = get_k_array(n_deg_sites, ratio)
			sim_data, first_passage_time = simulate_tf_search(sim_time, max_time, y0, k_array)
			first_passage[i,j] = first_passage_time
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, 3, 0)
			mean_occupancy_flanks[i, j] = compute_mean_occupancy(sim_data, 4, 0)
			mean_occupancy_local[i, j] = compute_mean_occupancy(sim_data, 3, 0, 4)

	np.save('simulation_output/first_passage_' + run_num + '.npy', first_passage)
	# np.save(target + '/simulation_output/first_passage_local_' + run_num + '.npy', first_passage_local)
	np.save('simulation_output/mean_occupancy_mot_' + run_num + '.npy', mean_occupancy_mot)
	np.save('simulation_output/mean_occupancy_flanks_' + run_num + '.npy', mean_occupancy_flanks)
	np.save('simulation_output/mean_occupancy_local_' + run_num + '.npy', mean_occupancy_local)


# parses input and runs appropriate simulation or sensitivity analysis
def main():
	parser = argparse.ArgumentParser(description='Get run number and sensitivity analysis target')
	parser.add_argument('run_num', type=str, help='run number')
	parser.add_argument('target', type=str, help='sensitivity analysis target variable')
	parser.add_argument("--y0", nargs="+", default=[1000, 0, 0, 0, 1, 100])
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
	if args.target == 'core_kinetics':
		core_kinetics_sensitivity(args.target, args.run_num, args.y0, factor)
	if args.target == 'flank_kinetics':
		flank_kinetics_sensitivity(args.target, args.run_num, args.y0, factor)
	if args.target == 'n_TF':
		n_tf_sensitivity(args.target, args.run_num, args.y0, factor)
	if args.target == 'sliding_kinetics':
		sliding_kinetics_sensitivity(args.target, args.run_num, args.y0, factor)
	if args.target == 'diffusion_kinetics':
		diffusion_kinetics_sensitivity(args.target, args.run_num, args.y0, factor)
	if args.target == 'DNA_concentration':
		dna_concentration_sensitivity(args.target, args.run_num, args.y0, factor)

	print('Run completed')


if __name__ == "__main__":
	main()
                                            