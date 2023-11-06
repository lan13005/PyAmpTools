
This is a rate test to compare MCMC sample rates to Bootstrap iteration rate
between emcee and amptools. This is probably the closest comparison we can make
on speed between the two methodologies.

benchmark.py runs fit.py (for BS MLE fits) and benchmark_mcmc.py for MCMC results.
A slurm submit script is available to run in batch, including a submit script to run
  an array of jobs.

benchmark_mcmc.py is a copy of mcmc.py example to keep the main one clean

3 tests are run on gluex data:
1. Mass independent fit in the a2(1320) peak of etapi data
2. Mass dependent fit in the lowest t-bin used in the a2 differential xsec
3. rho sdme fit in 5th t-bin

Results:
<img width="60%" src="benchmark_mcmc_vs_mle.png" alt="mcmc_vs_mle png" />
