# STR Kinetics Gillespie Simulation
![image](state_diagram.png)
## Scaling up computation
1) Parallelizing jobs to run sensitivity analysis
  - Could rewrite gillespie to go to just first TF bound to motif to get time to first passage and that would significantly cut down on time but would not get the metrics for mean occupancy etc.
  - Could also rewrite gillespie to go to first passage and if that is shorter than the total amount of time, then finish the simulation time. This makes it so we can just run gillespie once instead of max three times and ensure that we get to first passage once. Can calculate mean occupancy etc. just using the sim data within the simulation time range. This runs the danger though of running for an absurdly long time for parameters that will never lead to motif binding, so we have to decide how to average values for time to first passage when it is too large to run the simulation.
  
## Fixing reaction rates & assumptions
1) TF-DNA 2D sliding
2) Kd_in_out should be the ratio between the antenna and nucleus volume
3) Antenna volume
- This volume ratio of 1/100th of nucleus volume does not match up with the number of BP in the antenna. Two ways of thinking about this: radius of gyration of DNA, how much DNA occupies a certain volume. 
4) Kinetics of TF-DNA between random and repeats
- Need to update to use the ratio from the 2s kinetic measurements plot. ~0.08 1/Ms off rate for motif+repeats and ~0.095 1/Ms off rate for motif+random.
5) Multipling by the concentration of DNA in certain gillespie reactions
- This feels fine to do.
