# STR Kinetics Gillespie Simulation
![image](state_diagram.png)
## Scaling up computation
1) Parallelizing jobs to run sensitivity analysis
  - Could rewrite gillespie to go to just first TF bound to motif to get time to first passage and that would significantly cut down on time but would not get the metrics for mean occupancy etc.
  - Could also rewrite gillespie to go to first passage and if that is shorter than the total amount of time, then finish the simulation time. This makes it so we can just run gillespie once instead of max three times and ensure that we get to first passage once. Can calculate mean occupancy etc. just using the sim data within the simulation time range. This runs the danger though of running for an absurdly long time for parameters that will never lead to motif binding, so we have to decide how to average values for time to first passage when it is too large to run the simulation.

## Running sensitivity analysis
1) Ssh into sherlock. Clone this repository using `git clone`. Run `ml python/3.6.1`.
2) Run a the simulation using the following command: `python3 parallel_sensitivity_analysis.py n_runs simulation_type`. `n_runs` indicates how many jobs you want to submit or how many times you want to run the simulation. `simulation_type` indicates what to run, whether the baseline simulation, indicated with `simulation` passed in, or a sensitivity analysis, which can be one of the following: `n_flanks`, `core_affinity`, `core_kinetics`, `flank_kinetics`, `n_TF`, `sliding_kinetics`, `diffusion_kinetics`, `DNA_concentration`. For each analysis you do, make sure to create a folder titled with the `simulation_type` you are calling as well as a folder titled `simulation_output` within it. In addition, create a folder titled `txt_output` which is where the outputs from all the jobs will be saved.
3) Check on the status, whether the jobs finished and if there were any errors by calling `cat txt_output/*`.
4) Process the simulation using the following command: `python3 sensitivity_processing.py n_runs simulation_type`.

