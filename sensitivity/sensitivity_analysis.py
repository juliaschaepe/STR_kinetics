
import numpy as np
import argparse


def get_k_array(n_deg_sites, factor, DNA=5e-5, core_affinity=1e-7, core_koff=0.015,
				flank_koff=0.02, velocity_prob=1000 / 7, Kd_in_out=1e4,
				tf_diffusion=10e-7 / 100):
	nuc_vol = 3e-15
	ant_vol = 3.3e-17
	D = tf_diffusion  # cm^2/s --> L/s
	R = np.cbrt(3 * ant_vol / (4 * np.pi))

	Kd_23 = core_affinity  # source: MITOMI data
	Kd_24 = core_affinity / (factor * n_deg_sites)
	Kd_34 = Kd_24 / Kd_23
	Kd_12 = ((1 + DNA / Kd_23 + DNA / Kd_24) * Kd_in_out)

	k12 = 4 * np.pi * D * R / nuc_vol
	k21 = k12 * Kd_12  # to maintain "detailed balance"

	k32 = core_koff  # source: MITOMI data - this is koff
	k23 = k32 / Kd_23  # detailed balance - this is kon

	k42 = flank_koff  # source: MITOMI data - this is koff
	k24 = k42 / Kd_24  # detailed balance - this is kon

	k43 = velocity_prob / (n_deg_sites / 2)  # this might need to be re-evaluated
	k34 = k43 / Kd_34  # detailed balance

	return (k12, k21, k23, k24, k32, k34, k42, k43)


def simulate_TFsearch(T, y0, k_array, DNA=5e-5, frac_DNA=1 / 10):
	t = 0
	k12, k21, k23, k24, k32, k34, k42, k43 = k_array
	n_tot = np.sum(y0)
	i = 0
	y = y0
	n_rows = 100000
	sim_data = np.zeros((n_rows, len(y0) + 2))
	w_mapping = [([0], [1]),
				 ([1], [0]), ([1, 4], [2]), ([1, 5], [3]),
				 ([2], [1, 4]), ([2, 5], [3, 4]),
				 ([3], [1, 5]), ([3, 4], [2, 5])]

	while t < T:

		# calculates likelihood of each reaction
		w12 = k12 * y[0] * DNA
		w21 = k21 * y[1] * DNA
		# this assumes the concentration of DNA in the antenna is 1/100 of all DNA
		w23 = k23 * y[1] * y[4] * DNA
		w32 = k32 * y[2]
		w24 = k24 * y[1] * y[5] * DNA
		w42 = k42 * y[3]
		w34 = k34 * y[2]
		w43 = k43 * y[3]
		# this ensures conservation of 1 motif and 100 deg sites without messing with rates
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

	np.array(sim_data)
	return sim_data[:np.argmax(sim_data[:, 1]) + 1, :]

# computes the fraction of time that the target is occupied
def compute_fraction_time_occupied(simulation_data, target_idx, dt_index, time_index, target_idx_2 = None):
    target_data = simulation_data[:-1,target_idx]
    if target_idx_2 is not None:
        target_data = np.add(target_data, simulation_data[:-1, target_idx_2])
    tot_occupied_time = np.sum(simulation_data[(np.where(target_data > 0)[0] + 1),dt_index])
    return tot_occupied_time/simulation_data[-1,time_index]


# computes the average occupancy of the target
def compute_mean_occupancy(simulation_data, target_idx, dt_index, target_idx_2 = None):
    if target_idx_2 is not None:
        return np.average(np.add(simulation_data[:-1,target_idx], simulation_data[:-1,target_idx_2]), weights = simulation_data[1:, dt_index])
    return np.average(simulation_data[:-1,target_idx], weights = simulation_data[1:, dt_index])


def save_output(target, run_num, first_passage, mean_occupancy_mot, mean_occupancy_deg, mean_occupancy_ant):
    np.save('simulation_output/' + target + '_sensitivity_first_passage_' + run_num + '.npy', first_passage)
    np.save('simulation_output/' + target + '_sensitivity_mean_occupancy_mot_' + run_num + '.npy', mean_occupancy_mot)
    np.save('simulation_output/' + target + '_sensitivity_mean_occupancy_deg_' + run_num + '.npy', mean_occupancy_deg)
    np.save('simulation_output/' + target + '_sensitivity_mean_occupancy_ant_' + run_num + '.npy', mean_occupancy_ant)


def initialize_storage(n_factor, n_var):
    return np.zeros((n_factor, n_var)), np.zeros((n_factor, n_var)), np.zeros((n_factor, n_var)), np.zeros((n_factor, n_var))


def n_deg_sites_sensitivity(target, run_num):
	factor = np.geomspace(1e-3, 1, num=10)
	n_deg = np.array([1, 10, 100, 500, 1000])
	first_passage, mean_occupancy_mot, mean_occupancy_deg, mean_occupancy_ant = initialize_storage(len(factor), len(n_deg))
	for i, ratio in enumerate(factor):
		for j, n in enumerate(n_deg):
			y0 = [500, 0, 0, 0, 1, n]
			k_array = get_k_array(n, ratio)
			sim_data = simulate_TFsearch(1e3, y0, k_array)
			mean_occupancy_mot[i, j] = compute_mean_occupancy(sim_data, 3, 0)
			mean_occupancy_deg[i, j] = compute_mean_occupancy(sim_data, 4, 0)
			mean_occupancy_ant[i, j] = compute_mean_occupancy(sim_data, 3, 0, 4)
			try:
				motif_bound = np.where(sim_data[:, 3] > 0)[0]
				first_passage[i, j] = sim_data[motif_bound[0], 0]
			except:
				continue
	save_output(target, run_num, first_passage, mean_occupancy_mot, mean_occupancy_deg, mean_occupancy_ant)



def main():
	parser = argparse.ArgumentParser(description='Get run number and sensitivity analysis target')
	parser.add_argument('run_num', type=str, help='run number')
	parser.add_argument('target', type=str, help='sensitivity analysis target variable')
	args = parser.parse_args()

	# options for targets: 'n_deg_sites', 'core_affinity', 'core_kinetics', 'flank_kinetics', 'n_TF'
	# 'sliding_kinetics', 'diffusion_kinetics', 'DNA_concentration'

	if args.target == 'n_deg_sites':
		n_deg_sites_sensitivity(args.target, args.run_num)
	# these options need to be implemented
	# if args.target == 'core_affinity':
	# 	core_affinity_sensitivity(args.run_num)
	# if args.target == 'core_kinetics':
	# 	core_kinetics_sensitivity(args.run_num)
	# if args.target == 'flank_kinetics':
	# 	flank_kinetics_sensitivity(args.run_num)
	# if args.target == 'n_TF':
	# 	n_TF_sensitivity(args.run_num)
	# if args.target == 'sliding_kinetics':
	# 	sliding_kinetics_sensitivity(args.run_num)
	# if args.target == 'diffusion_kinetics':
	# 	diffusion_kinetics_sensitivity(args.run_num)
	# if args.target == 'DNA_concentration':
	# 	DNA_concentration_sensitivity(args.run_num)

	print('Run completed')




if __name__ == "__main__":
    main()
