Benchmark scaling of fit and mcmc as a function of MPI (GPU) processes.

These tests are performed using [1, 2, 4, 8, 16] T4 Nvidia GPUs on a single node. The rho-SDME dataset is used. Overhead / cpu usage becomes significant after around 4 GPUs for this study. More complicated fits would push this out further

<img width="60%" src="./results/scaling.png" alt="mpi_scaling png" />
