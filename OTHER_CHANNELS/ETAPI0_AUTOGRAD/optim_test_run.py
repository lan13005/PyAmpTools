import os
import numpy as np
import subprocess
from multiprocessing import Pool
import sys
from itertools import product
import time
from tqdm import tqdm

if not os.path.exists("COMPARISONS"):
    os.makedirs("COMPARISONS")

def run_command(job_info):
    job_idx, bins, optimizer, setting = job_info
    cmd = f"python optim_test.py pyamptools_mc.yaml --method {optimizer} --bins {' '.join([str(x) for x in bins])} --setting {setting} > COMPARISONS/mle_{job_idx}.log"
    print(cmd)
    # print(f"Starting job {job_idx} for optimizer {optimizer} and bins: {bins}")
    try:
        process = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return job_idx, optimizer, True, process.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error in job {job_idx} with optimizer {optimizer}: {e}", file=sys.stderr)
        return job_idx, optimizer, False, e.stderr

n_bins = 1 # 34
# optimizers = ['minuit-numeric', 'L-BFGS-B', 'trust-ncg', 'trust-krylov']
# NOTE: currently trust region methods break parameter bounds which raises an error
optimizers = ['minuit-analytic']
max_concurrent = min(1, n_bins)
settings = range(1) # range(30)

# Create job assignments for each optimizer
job_assignments = {}
job_counter = 0
for setting in settings:
    for optimizer in optimizers:
        for bin_idx in range(n_bins):
            job_assignments[job_counter] = ([bin_idx], optimizer, setting)
            job_counter += 1

total_jobs = len(job_assignments)

print(f"Total jobs: {total_jobs}")

# Use Pool to manage concurrent processes
start_time = time.time()
with Pool(processes=max_concurrent) as pool:
    # Create list of job arguments
    job_args = [(job_idx, bins, optimizer, setting) for job_idx, (bins, optimizer, setting) in job_assignments.items()]
    
    # Map jobs to pool and process results as they complete
    with tqdm(total=total_jobs, desc="Processing jobs", unit="job") as pbar:
        for job_idx, optimizer, success, output in pool.imap_unordered(run_command, job_args):
            if success:
                print(f"Job {job_idx} with optimizer {optimizer} completed successfully")
            else:
                print(f"Job {job_idx} with optimizer {optimizer} failed: {output}", file=sys.stderr)
            pbar.update(1)

end_time = time.time()
print(f"Total time taken: {end_time - start_time} seconds for {total_jobs} jobs")