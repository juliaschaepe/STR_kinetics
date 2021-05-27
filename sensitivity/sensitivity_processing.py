import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import LogNorm
import seaborn as sns
import scipy

def initialize_storage(n_jobs, n_factor, n_var):
    return np.zeros((n_jobs, n_factor, n_var)), np.zeros((n_jobs, n_factor, n_var)), np.zeros((n_jobs, n_factor, n_var)), np.zeros((n_jobs, n_factor, n_var))

def load_arrays(num_jobs, target):
	example = np.load(target + '/simulation_output/' + target + '_sensitivity_first_passage_0.npy')
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(num_jobs, example.shape[0], example.shape[1])
	for run_num in range(num_jobs):
		first_passage[run_num, :, :] = np.load(target + '/simulation_output/' + target + '_sensitivity_first_passage_' + str(run_num) + '.npy')
		mean_occupancy_mot[run_num, :, :] = np.load(target + '/simulation_output/' + target + '_sensitivity_mean_occupancy_mot_' + str(run_num) + '.npy')
		mean_occupancy_flanks[run_num, :, :] = np.load(target + '/simulation_output/' + target + '_sensitivity_mean_occupancy_flanks_' + str(run_num) + '.npy')
		mean_occupancy_local[run_num, :, :] = np.load(target + '/simulation_output/' + target + '_sensitivity_mean_occupancy_local_' + str(run_num) + '.npy')
	return first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local


def sensitivity_plot(target, num_jobs, factor, run_vars, first_passage, mean_occupancy_mot, mean_occupancy_flanks):
	fig = plt.figure(figsize=(20, 6))
	factor = np.around(factor, 3)
	ax = fig.add_subplot(131)
	sns.heatmap(np.mean(first_passage, axis=0), norm=LogNorm(), xticklabels=run_vars)
	ax.set_ylabel('affinity ratio\n(flanks/core)', fontsize=18)
	ax.set_xlabel(target, fontsize=18)
	ax.set_xticklabels(run_vars, rotation=45)
	ax.set_yticklabels(factor, rotation=0)
	ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
	ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
	ax.tick_params(labelsize=14)
	ax.set_title('First passage time', fontsize=20)

	ax = fig.add_subplot(132)
	sns.heatmap(np.mean(mean_occupancy_mot, axis=0), norm=LogNorm(), xticklabels=run_vars)
	ax.set_ylabel('affinity ratio\n(flanks/core)', fontsize=18)
	ax.set_xlabel(target, fontsize=18)
	ax.set_yticklabels(factor, rotation=0)
	ax.set_xticklabels(run_vars, rotation=45)
	ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
	ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
	ax.tick_params(labelsize=14)
	ax.set_title('Mean consensus occupancy', fontsize=20)

	ax = fig.add_subplot(133)
	sns.heatmap(np.mean(mean_occupancy_flanks, axis=0), norm=LogNorm(), xticklabels=run_vars)
	ax.set_ylabel('affinity ratio\n(flanks/core)', fontsize=18)
	ax.set_xlabel(target, fontsize=18)
	ax.set_yticklabels(factor, rotation=0)
	ax.set_xticklabels(run_vars, rotation=45)
	ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
	ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
	ax.tick_params(labelsize=14)
	ax.set_title('Mean flanks occupancy', fontsize=20)
	plt.tight_layout()
	plt.savefig(target + '/' + target + '_heatmaps.pdf', dpi=300)
	# plt.show()

	fig = plt.figure()
	ax = fig.add_subplot(111)
	spearman_arr = np.zeros(len(run_vars))
	pearson_arr = np.zeros(len(run_vars))
	for i, n in enumerate(run_vars):
		r, p = scipy.stats.spearmanr(-np.hstack([factor] * num_jobs), first_passage[:, :, i].flat)
		spearman_arr[i] = r
		r, p = scipy.stats.pearsonr(-np.hstack([factor] * num_jobs), first_passage[:, :, i].flat)
		pearson_arr[i] = r

	ax.plot(run_vars, spearman_arr, marker='o', ls='')
	ax.plot(run_vars, pearson_arr, marker='o', ls='')
	ax.legend(['Spearman rho', 'Pearson R'])
	ax.set_xscale('log')
	ax.set_xlabel(target, fontsize=18)
	ax.set_ylabel('correlation', fontsize=18)
	ax.set_title('Correlation between ' + target + ' and time to first passage')
	ax.tick_params()
	fig.tight_layout()
	plt.savefig(target + '/' + target + '_correlation.pdf', dpi=300)


def get_run_vars(target):
	if target == 'simulation':
		# TODO: this is not right
		return np.array([1, 10, 100, 500, 1000])
	if target == 'n_flanks':
		return np.array([1, 10, 100, 500, 1000])
	if target == 'core_affinity':
		return np.geomspace(1e-10, 1e-6, 10)
	if target == 'core_kinetics':
		return np.geomspace(1e-3, 1, 10)
	if target == 'flank_kinetics':
		return np.geomspace(1e-2, 1, 10)
	if target == 'n_TF':
		return np.array([50, 100, 200, 500, 1000, 2000, 5000])
	if target == 'sliding_kinetics':
		return np.geomspace(5, 1000, 10)
	if target == 'diffusion_kinetics':
		# TODO: this needs to be updated
		return np.geomspace(1e-5,1e-2, 10)
	if target == 'DNA_concentration':
		return np.geomspace(1e-7, 1e-3, 10)


def main():
	parser = argparse.ArgumentParser(description='Get number of jobs and sensitivity analysis target')
	parser.add_argument('num_jobs', type=int, help='number of jobs')
	parser.add_argument('target', type=str, help='sensitivity analysis target variable')
	args = parser.parse_args()

	# options for targets: 'n_flanks_sites', 'core_affinity', 'core_kinetics', 'flank_kinetics', 'n_TF'
	# 'sliding_kinetics', 'diffusion_kinetics', 'DNA_concentration'
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = load_arrays(args.num_jobs, args.target)
	factor = np.geomspace(1e-3, 1, num=10)

	run_vars = get_run_vars(args.target)
	sensitivity_plot(args.target, args.num_jobs, factor, run_vars, first_passage, mean_occupancy_mot, mean_occupancy_flanks)

if __name__ == "__main__":
    main()