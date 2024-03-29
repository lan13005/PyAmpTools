import os
import matplotlib.pyplot as plt
import numpy as np
from utils import setPlotStyle
setPlotStyle()

### GLOBALS
ngpus = np.array([1, 2, 4, 8, 16])
nprocs = [ngpu+1 for ngpu in ngpus]
sample = "SDME_EXAMPLE/sdme.cfg"

def run_test(program, args=""):
    for nproc, ngpu in zip(nprocs, ngpus):
        tag = program.split(".")[0]
        cmd  = f"mpiexec -np {nproc} python $REPO_HOME/src/pyamptools/{program} $REPO_HOME/tests/samples/{sample} {args}"
        cmd += f" >> results/{tag}_nproc{ngpu}.log 2>&1"

        print(cmd)
        os.system(cmd)

        if program == "mle.py":
            os.system(f'rm -f result_0.fit seed_0.txt')

# #################### RUN TESTS ON mle.py ####################
# program = "mle.py"
# for nproc, ngpu in zip(nprocs, ngpus):
#     cmd  = f"mpiexec -np {nproc} python $REPO_HOME/src/pyamptools/mle.py $REPO_HOME/tests/samples/{sample}"
#     cmd += f" >> results/fit_nproc{ngpu}.log 2>&1"
#     print(cmd)
#     os.system(cmd)
#     os.system(f'rm -f result_0.fit seed_0.txt')

# # #################### RUN TESTS ON mcmc.py ####################
# program = "mcmc.py"
# args = f" --burnin 0 --nsamples 100 --overwrite"
# for nproc, ngpu in zip(nprocs, ngpus):
#     cmd  = f"mpiexec -np {nproc} python $REPO_HOME/src/pyamptools/mcmc.py --cfgfiles $REPO_HOME/tests/samples/{sample} {args}"
#     cmd += f" >> results/mcmc_nproc{ngpu}.log 2>&1"
#     print(cmd)
#     os.system(cmd)
#     os.system(f'rm -f mcmc')

#################### DRAW RESULTS ####################
ngpus = np.array([1, 2, 4, 8, 16])
fit_times = []
mcmc_times = []
for ngpu in ngpus:
    for tag, times in zip(["fit", "mcmc"], [fit_times, mcmc_times]):
        fit_files = f"results/{tag}_nproc{ngpu}.log"
        with open(fit_files, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Fit time:"):
                    time = float(line.split()[2])
                    times.append(time)

fit_times = np.array(fit_times)
mcmc_times = np.array(mcmc_times)
fit_times /= fit_times[0]
mcmc_times /= mcmc_times[0]

fig = plt.figure(figsize=(6,4))

plt.plot(ngpus, fit_times,  "o-",  c='royalblue',   markerfacecolor='none', markeredgewidth=2, label="mle.py scaling")
plt.plot(ngpus, mcmc_times, "o--", c='orange', markerfacecolor='none', markeredgewidth=2, label="mcmc.py scaling")

f_gpus = np.linspace(ngpus.min(), ngpus.max(), 100)
f_times = 1 / f_gpus
plt.plot(ngpus, 1/ngpus, '--', c='black', label="Ideal scaling")

plt.ticklabel_format(axis='y', style='plain')
plt.ticklabel_format(axis='x', style='plain')

plt.ylim(0.05,1)
plt.xlim(ngpus.min(), ngpus.max()+1)

plt.yscale("log", base=2)
plt.xscale("log", base=2)

plt.xlabel("Number of GPUs")
plt.ylabel("Run Time (Normalized)")
plt.legend()
plt.tight_layout()

plt.savefig("results/scaling.png")
