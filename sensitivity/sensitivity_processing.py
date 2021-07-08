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
	example = np.load(target + '/simulation_output/' + target + '_sensitivity_first_passage_1.npy')
	first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = initialize_storage(num_jobs, example.shape[0], example.shape[1])
	for run_num in range(num_jobs):
		first_passage[run_num, :, :] = np.load(target + '/simulation_output/' + target + '_sensitivity_first_passage_' + str(run_num) + '.npy')
		mean_occupancy_mot[run_num, :, :] = np.load(target + '/simulation_output/' + target + '_sensitivity_mean_occupancy_mot_' + str(run_num) + '.npy')
		mean_occupancy_flanks[run_num, :, :] = np.load(target + '/simulation_output/' + target + '_sensitivity_mean_occupancy_flanks_' + str(run_num) + '.npy')
		mean_occupancy_local[run_num, :, :] = np.load(target + '/simulation_output/' + target + '_sensitivity_mean_occupancy_local_' + str(run_num) + '.npy')
	return first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local


def sensitivity_plot(target, num_jobs, factor, run_vars, first_passage, mean_occupancy_mot, mean_occupancy_flanks, log):
	fig = plt.figure(figsize=(20, 6))
	factor = np.around(factor, 3)
	run_vars = np.asarray(run_vars)
	ax = fig.add_subplot(131)
	if log:
		sns.heatmap(np.mean(first_passage, axis=0), xticklabels=run_vars, norm=LogNorm())
	else:
		sns.heatmap(np.mean(first_passage, axis=0), xticklabels=run_vars)
	ax.set_ylabel('affinity ratio\n(flanks/core)', fontsize=18)
	ax.set_xlabel(target, fontsize=18)
	run_vars_ticks = run_vars
	factor_ticks = factor
	run_vars_ticks = [format(x, ".1e") for x in run_vars]
	factor_ticks = [format(y, ".1e") for y in factor]
	ax.set_xticklabels(run_vars_ticks, rotation=30)
	ax.set_yticklabels(factor_ticks, rotation=0)
	ax.tick_params(labelsize=14)
	ax.set_title('First passage time', fontsize=20)

	ax = fig.add_subplot(132)
	if log:
		sns.heatmap(np.mean(mean_occupancy_mot, axis=0), xticklabels=run_vars, norm=LogNorm())
	else:
		sns.heatmap(np.mean(mean_occupancy_mot, axis=0), xticklabels=run_vars)
	ax.set_ylabel('affinity ratio\n(flanks/core)', fontsize=18)
	ax.set_xlabel(target, fontsize=18)
	ax.set_yticklabels(factor_ticks, rotation=0)
	ax.set_xticklabels(run_vars_ticks, rotation=30)
	ax.tick_params(labelsize=14)
	ax.set_title('Mean motif occupancy', fontsize=20)

	ax = fig.add_subplot(133)
	if log:
		sns.heatmap(np.mean(mean_occupancy_flanks, axis=0), xticklabels=run_vars, norm=LogNorm())
	else:
		sns.heatmap(np.mean(mean_occupancy_flanks, axis=0), xticklabels=run_vars)
	ax.set_ylabel('affinity ratio\n(flanks/core)', fontsize=18)
	ax.set_xlabel(target, fontsize=18)
	ax.set_yticklabels(factor_ticks, rotation=0)
	ax.set_xticklabels(run_vars_ticks, rotation=30)
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


def simulation_plot(target, run_num, factor):
	first_passage = np.zeros((run_num, len(factor)))
	mean_occupancy_mot = np.zeros((run_num, len(factor)))
	mean_occupancy_flanks = np.zeros((run_num, len(factor)))
	mean_occupancy_local = np.zeros((run_num, len(factor)))
	for i in range(run_num):
		first_passage[i,:] = np.load(target + '/simulation_output/first_passage_' + str(i) + '.npy')
		mean_occupancy_mot[i,:] = np.load(target + '/simulation_output/mean_occupancy_mot_' + str(i) + '.npy')
		mean_occupancy_flanks[i,:] = np.load(target + '/simulation_output/mean_occupancy_flanks_' + str(i) + '.npy')
		mean_occupancy_local[i,:] = np.load(target + '/simulation_output/mean_occupancy_local_' + str(i) + '.npy')

	fig = plt.figure(figsize=(18, 10))
	factor = np.geomspace(1e-4, 10)
	ax = fig.add_subplot(111)
	l1 = ax.errorbar(x=factor, y=np.median(first_passage, axis=0),
					 yerr=np.std(first_passage, axis=0) / np.sqrt(run_num), marker='o',
					 color='k')
	ax.set_xscale('log')
	# ax.set_xlim(9e-4)
	# ax.set_ylim(-50, 2000)
	ax.set_ylabel('time (sec)', fontsize=20)
	ax.set_xlabel('ratio of affinity (flanks/core)', fontsize=20)
	ax.tick_params(labelsize=14)

	ax1 = ax.twinx()
	l2 = ax1.errorbar(x=factor, y=np.median(mean_occupancy_mot, axis=0),
					  yerr=np.std(mean_occupancy_mot, axis=0) / np.sqrt(run_num), marker='o',
					  color='C1')
	l3 = ax1.errorbar(x=factor, y=np.median(mean_occupancy_local, axis=0),
					  yerr=np.std(mean_occupancy_local, axis=0) / np.sqrt(run_num), marker='o',
					  color='C0')
	l4 = ax1.errorbar(x=factor, y=np.median(mean_occupancy_flanks, axis=0),
					  yerr=np.std(mean_occupancy_flanks, axis=0) / np.sqrt(run_num), marker='o',
					  color='C2')

	# highlight repeat and random regions for affinity ratios
	l5 = plt.axvspan(9e-3, 11e-3, color='black', alpha=0.3)
	l6 = plt.axvspan(9e-2, 11e-2, color='red', alpha=0.3)
	ax1.set_ylabel('mean occupancy (# TFs)', fontsize=20)
	ax1.tick_params(labelsize=14)

	ax1.legend((l1, l2, l3, l4, l5, l6),
			   ['first passage', 'mean occupancy on motif', 'mean occupancy in local volume',
				'mean occupancy on flanks', 'random flank affinity ratio range',
				'repetitive flank affinity ratio range'], fontsize=20, bbox_to_anchor=(1.1, 1),
			   loc='upper left')

	fig.tight_layout()
	plt.savefig(target + '/simulation_results_' + str(run_num) + '.pdf', dpi=300)

def get_run_vars(target):
	if target == 'n_flanks':
		return np.array([1, 10, 100, 500, 1000])
	if target == 'core_affinity':
		return np.geomspace(1e-10, 1e-6, 10)
	if target == 'koff':
		return np.geomspace(0.1, 1, 10)
	if target == 'n_TF':
		return np.array([1, 5, 10, 50, 100])
	if target == 'sliding_kinetics':
		return np.geomspace(0.1, 1e3, 10)
	if target == 'diffusion_kinetics':
		return np.geomspace(1e-3, 1e2, 10)
	if target == 'DNA_concentration':
		return np.geomspace(1e-7, 1e-3, 10)


def main():
	parser = argparse.ArgumentParser(description='Get number of jobs and sensitivity analysis target')
	parser.add_argument('run_num', type=int, help='number of runs')
	parser.add_argument('target', type=str, help='sensitivity analysis target variable')
	parser.add_argument("-log", type=bool, default=False)
	args = parser.parse_args()

	# options for targets: 'n_flanks', 'core_affinity', 'core_kinetics', 'flank_kinetics', 'n_TF'
	# 'sliding_kinetics', 'diffusion_kinetics', 'DNA_concentration'

	if args.target == 'simulation':
		factor = np.geomspace(1e-4, 10)
		simulation_plot(args.target, args.run_num, factor)
	else:
		first_passage, mean_occupancy_mot, mean_occupancy_flanks, mean_occupancy_local = load_arrays(
			args.run_num, args.target)
		factor = np.geomspace(1e-3, 1, num=10)
		run_vars = get_run_vars(args.target)
		sensitivity_plot(args.target, args.run_num, factor, run_vars, first_passage, mean_occupancy_mot, mean_occupancy_flanks, args.log)

if __name__ == "__main__":
    main()