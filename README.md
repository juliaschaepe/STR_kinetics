![image](state_diagram.png)

## Running a simulation or a sensitivity analysis
1) Ssh into sherlock. Clone this repository using `git clone`. Run `ml python/3.6.1`.
2) Run a the simulation using the following command: `python3 parallel_sensitivity_analysis.py n_runs simulation_type`. `n_runs` indicates how many jobs you want to submit or how many times you want to run the simulation. `simulation_type` indicates what to run, whether the baseline simulation, indicated with `simulation` passed in, or a sensitivity analysis, which can be one of the following: `n_flanks`, `core_affinity`, `core_kinetics`, `flank_kinetics`, `n_TF`, `sliding_kinetics`, `diffusion_kinetics`, `DNA_concentration`. For each analysis you do, make sure to create a folder titled with the `simulation_type` you are calling as well as a folder titled `simulation_output` within it. In addition, create a folder titled `txt_output` which is where the outputs from all the jobs will be saved.
3) Check on the status, whether the jobs finished and if there were any errors by calling `cat txt_output/*`.
4) Process the simulation using the following command: `python3 sensitivity_processing.py n_runs simulation_type`.

## Example output

![image](sensitivity/simulation_tfs_flanks.pdf)
![image](sensitivity/simulation_tfs_motif.pdf)
![image](sensitivity/simulation_tfs_local.pdf)

