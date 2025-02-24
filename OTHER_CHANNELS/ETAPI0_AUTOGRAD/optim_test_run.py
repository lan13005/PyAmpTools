import os
import numpy as np
import subprocess
from multiprocessing import Pool
import sys
from itertools import product
import time

def run_command(job_info):
    job_idx, bins, optimizer = job_info
    cmd = (
        # f"mpiexec -v -n 2 "
        # f"bash -c 'python optim_test.py --optimizer {optimizer} --bins {' '.join([str(x) for x in bins])}'"
        f"python optim_test.py --optimizer {optimizer} --bins {' '.join([str(x) for x in bins])}"
    )
    print(cmd)
    print(f"Starting job {job_idx} for optimizer {optimizer} and bins: {bins}")
    try:
        process = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return job_idx, optimizer, True, process.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error in job {job_idx} with optimizer {optimizer}: {e}", file=sys.stderr)
        return job_idx, optimizer, False, e.stderr

n_bins = 34
# Add trust-ncg as an optimizer option
# optimizers = ['minuit_numeric', 'minuit_analytic', 'lbfgs', 
# optimizers = ['L-BFGS-B']
optimizers = ['trust-ncg', 'trust-krylov']
max_concurrent = 20

# Create job assignments for each optimizer
job_assignments = {}
job_counter = 0
for optimizer in optimizers:
    for bin_idx in range(n_bins):
        job_assignments[job_counter] = ([bin_idx], optimizer)
        job_counter += 1

total_jobs = len(job_assignments)
assert sum(len(bins) for bins, _ in job_assignments.values()) == n_bins * len(optimizers), \
    "sum of bins does not match total number of bins across all optimizers"
    
# Use Pool to manage concurrent processes
start_time = time.time()
with Pool(processes=max_concurrent) as pool:
    # Create list of job arguments
    job_args = [(job_idx, bins, optimizer) for job_idx, (bins, optimizer) in job_assignments.items()]
    
    # Map jobs to pool and process results as they complete
    for job_idx, optimizer, success, output in pool.imap_unordered(run_command, job_args):
        if success:
            print(f"Job {job_idx} with optimizer {optimizer} completed successfully")
        else:
            print(f"Job {job_idx} with optimizer {optimizer} failed: {output}", file=sys.stderr)

end_time = time.time()
print(f"Total time taken: {end_time - start_time} seconds for {total_jobs} jobs")